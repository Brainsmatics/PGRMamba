"""Training with the original GCA loss and physical 6 um NMS/matching.
XYZ labels remain voxel coordinates; spacing defaults to (1,1,2) um/voxel.
Uses the currently active Model_net architecture, optimizer and loss formulas.
Training/validation GCA parameters remain (1.5, 6)/(1, 1), including the
original prediction-only coordinate scaling. These are not physical radii.
Timestamped logs are never cleared on import.
"""
import argparse
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from load_data import dataset0, dataset0_collate
from get_loss import GCA_Loss
import torch.backends.cudnn as cudnn
from torch import optim
import os
from tensorboardX import SummaryWriter
import numpy as np

from evaluate_prf1_csv_um import PhysicalDistanceEvaluator
# from predict0 import detection

def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)


def decode_output(pred, conf_thres=0.5):
    pred = pred.detach().cpu()
    pred_x, pred_y, pred_z = torch.sigmoid(pred[0]), torch.sigmoid(pred[1]), torch.sigmoid(pred[2])
    conf = torch.sigmoid(pred[3])

    d, h, w = pred_x.shape
    grid_h = torch.arange(h).view(1, h, 1).expand(d, h, w)  
    grid_w = torch.arange(w).view(1, 1, w).expand(d, h, w)  
    grid_d = torch.arange(d).view(d, 1, 1).expand(d, h, w)

    x = (pred_x + grid_h) * 4.0
    y = (pred_y + grid_w) * 4.0
    z = (pred_z + grid_d) * 4.0

    keep = conf > conf_thres
    if not torch.any(keep):
        return np.empty((0, 4), dtype=np.float32)

    coords_conf = torch.stack([x[keep], y[keep], z[keep], conf[keep]], dim=1).numpy().astype(np.float32)
    return coords_conf


def nms_distance_um(dets: np.ndarray, distance_thres_um: float = 6.0,
                    spacing_xyz=(1.0, 1.0, 2.0)) -> np.ndarray:
    geometry = PhysicalDistanceEvaluator(distance_thres_um, spacing_xyz)
    dets = np.asarray(dets, dtype=np.float32)
    if dets.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if dets.ndim != 2 or dets.shape[1] != 4 or not np.isfinite(dets).all():
        raise ValueError('NMS input must be finite (N,4) x,y,z,confidence detections')
    order = np.argsort(-dets[:, 3])
    xyz_um = dets[:, :3].astype(np.float64) * geometry.spacing_xyz
    kept = []
    while len(order):
        index = order[0]
        kept.append(index)
        rest = order[1:]
        distances = np.linalg.norm(xyz_um[rest] - xyz_um[index], axis=1)
        order = rest[distances > distance_thres_um]
    return dets[kept]


def extract_gt_coords(label_batch, batch_index=0):
    gt_coords = np.empty((0, 3), dtype=np.float32)

    # dataset0_collate: bboxes = np.array(bboxes) 
    if isinstance(label_batch, np.ndarray) and label_batch.shape[0] > batch_index:
        gt0 = label_batch[batch_index]
        if gt0 is not None and np.asarray(gt0).size > 0:
            gt = np.asarray(gt0, dtype=np.float32)
            if gt.ndim == 2 and gt.shape[1] >= 3:
                gt_coords = gt[:, :3]
    elif isinstance(label_batch, (list, tuple)) and len(label_batch) > batch_index:
        gt0 = label_batch[batch_index]
        if gt0 is not None and len(gt0) > 0:
            gt = np.asarray(gt0, dtype=np.float32)
            if gt.ndim == 2 and gt.shape[1] >= 3:
                gt_coords = gt[:, :3]

    return gt_coords



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train neuron detection model')
    parser.add_argument('--seed', type=int, default=2026, help='Random seed (default: 2025)')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--profile_only', action='store_true', help='Report model complexity without training')
    parser.add_argument('--spacing_xyz', type=float, nargs=3, default=(1., 1., 2.))
    parser.add_argument('--distance_um', type=float, default=6., help='Physical NMS and matching radius only; does not change GCA loss')
    args = parser.parse_args()
    PhysicalDistanceEvaluator(args.distance_um, args.spacing_xyz)
    if args.epochs < 1:
        parser.error('--epochs must be positive')

    # Respect the GPU selection supplied by the shell; default to GPU 0 only
    # when CUDA_VISIBLE_DEVICES is absent.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
          '(visible GPUs are renumbered from logical cuda:0)')
    from feature_extract_net import Model_net
    set_random_seed(args.seed)
    print(f'[reproducibility] random seed = {args.seed}')

    Cuda = True
    batch_size = 1
    data0 = dataset0(src=r'/media/sdb1/czb/Neuron_Detection_new/3D_NSD/train') # path to training data
    validata0 = dataset0(src=r'/media/sdb1/czb/Neuron_Detection_new/3D_NSD/val') # path to validation data
    lr = 1e-2 # learning rate
    epoch = args.epochs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, 'model') # path to save model
    os.makedirs(save_path, exist_ok=True)
    from datetime import datetime
    result_dir = os.path.join(save_path, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    writer = SummaryWriter(result_dir)
    print(f'[physical distance] NMS / matching = {args.distance_um:g} um; spacing xyz={args.spacing_xyz}; GCA loss uses original settings')

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(args.seed)
    data_loader = DataLoader(
        dataset=data0,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=dataset0_collate,
        generator=data_loader_generator,
    )
    vali_data_loader = DataLoader(dataset=validata0, batch_size=1, collate_fn=dataset0_collate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Model_net([3, 3]).train()
    net = net.to(device)
    from model_complexity import report_model_complexity
    sample_image, _ = data0[0]
    complexity = report_model_complexity(
        net, (1, *sample_image.shape), os.path.join(result_dir, 'model_complexity.json'))
    del sample_image
    writer.add_text('model/complexity', str(complexity), 0)
    if args.profile_only:
        writer.close()
        raise SystemExit(0)
    net = torch.nn.DataParallel(net)

    base_lr = lr
    soma_lr = lr * 0.01      
    dog_lr = lr * 0.01       
    gate_lr = lr * 0.01      

    model_for_opt = net.module  

    soma_params, soma_param_ids = [], set()
    dog_params, dog_param_ids = [], set()
    gate_params, gate_param_ids = [], set()
    fusion_params, fusion_param_ids = [], set()

    for _, module in model_for_opt.named_modules():
        if module.__class__.__name__ == "SFE":
            for p in module.parameters(recurse=True):
                soma_params.append(p)
                soma_param_ids.add(id(p))
                
        elif module.__class__.__name__ == "RegMamba":
            if hasattr(module, 'dog_conv'):
                for p in module.dog_conv.parameters():
                    dog_params.append(p)
                    dog_param_ids.add(id(p))
                    
            if hasattr(module, 'gate_beta'):
                gate_params.append(module.gate_beta)
                gate_param_ids.add(id(module.gate_beta))

            if hasattr(module, 'gate_dog'):
                gate_params.append(module.gate_dog)
                gate_param_ids.add(id(module.gate_dog))

    all_special_ids = soma_param_ids.union(dog_param_ids).union(gate_param_ids).union(fusion_param_ids)
    other_params = [p for p in model_for_opt.parameters() if id(p) not in all_special_ids]

    param_groups = [
        {"params": other_params, "lr": base_lr},
    ]

    if soma_params:
        param_groups.append({"params": soma_params, "lr": soma_lr})
    if dog_params:
        param_groups.append({"params": dog_params, "lr": dog_lr})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": gate_lr})

    optimizer = optim.Adam(param_groups)

    print(f"[optimizer] Param groups created:")
    print(f"  - Base (Mamba/etc): lr={base_lr:.3e}, count={len(other_params)}")
    if soma_params: print(f"  - SFE: lr={soma_lr:.3e}, count={len(soma_params)}")
    if dog_params: print(f"  - RegMamba DoG Conv: lr={dog_lr:.3e}, count={len(dog_params)}")
    if gate_params: print(f"  - RegMamba Gate (alpha/beta): lr={gate_lr:.3e}, count={len(gate_params)}")

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    best_f1 = -1.0
    best_epoch = -1
    best_path = os.path.join(save_path, 'best_f1_model.pth')
    seed_best_path = os.path.join(save_path, f'best_f1_model_seed_{args.seed}.pth')

    for i in range(epoch):
        total_loss = 0
        total_loss_x, total_loss_y, total_loss_z, total_loss_conf = 0,0,0,0
        for image, label in data_loader:
            image = torch.tensor(image).cuda()
            outputs = net(image)
            # print(outputs)
            a = GCA_Loss(outputs, label, device, 1.5, 6)
            loss, loss_x, loss_y, loss_z, loss_conf = a.get_loss()
            # print(loss)
            total_loss += loss.data
            total_loss_x += loss_x.data
            total_loss_y += loss_y.data
            total_loss_z += loss_z.data
            total_loss_conf += loss_conf.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        writer.add_scalar('train/total_loss', total_loss, i)
        writer.add_scalar('train/total_loss_conf', total_loss_conf, i)
        writer.add_scalar('train/total_loss_x', total_loss_x, i)
        writer.add_scalar('train/total_loss_y', total_loss_y, i)
        writer.add_scalar('train/total_loss_z', total_loss_z, i)

        print('Epoch {} finished! Total loss is {}'.format(i, total_loss))
        if i % 10 == 0:
            torch.save(net.module.state_dict(), os.path.join(save_path, '{}.pth'.format(i)))

        
        net.eval()
        evaluator = PhysicalDistanceEvaluator(args.distance_um, args.spacing_xyz)
        val_total_loss = 0
        val_total_loss_x, val_total_loss_y, val_total_loss_z, val_total_loss_conf = 0, 0, 0, 0
        total_tp, total_fp, total_fn = 0, 0, 0
        with torch.no_grad():
            for image, label in vali_data_loader:
                image_t = torch.tensor(image).to(device)
                outputs = net(image_t)
                a = GCA_Loss(outputs, label, device, 1, 1)
                
                loss, loss_x, loss_y, loss_z, loss_conf = a.get_loss()
                val_total_loss += loss.data
                val_total_loss_x += loss_x.data
                val_total_loss_y += loss_y.data
                val_total_loss_z += loss_z.data
                val_total_loss_conf += loss_conf.data

                
                pred_coords_conf = decode_output(outputs[0], conf_thres=0.5)
                pred_coords_conf = nms_distance_um(pred_coords_conf, args.distance_um, args.spacing_xyz)
                pred_coords = pred_coords_conf[:, :3] if pred_coords_conf.size > 0 else np.empty((0, 3), dtype=np.float32)
                gt_coords = extract_gt_coords(label, batch_index=0)
                res = evaluator.match_predictions(pred_coords, gt_coords)
                total_tp += res.tp
                total_fp += res.fp
                total_fn += res.fn

        metrics = evaluator.compute_metrics(total_tp, total_fp, total_fn)
        writer.add_scalar('validation/validation_total_loss', val_total_loss, i)
        writer.add_scalar('validation/validation_total_loss_conf', val_total_loss_conf, i)
        writer.add_scalar('validation/validation_total_loss_x', val_total_loss_x, i)
        writer.add_scalar('validation/validation_total_loss_y', val_total_loss_y, i)
        writer.add_scalar('validation/validation_total_loss_z', val_total_loss_z, i)
        writer.add_scalar('validation/precision', metrics['precision'], i)
        writer.add_scalar('validation/recall', metrics['recall'], i)
        writer.add_scalar('validation/f1', metrics['f1'], i)

        is_best = metrics['f1'] > best_f1
        if is_best:
            best_f1 = float(metrics['f1'])
            best_epoch = int(i)
            torch.save(net.module.state_dict(), best_path)
            torch.save(net.module.state_dict(), seed_best_path)
            writer.add_scalar('validation/best_f1', best_f1, i)
            writer.add_text('validation/best', f'NEW BEST F1={best_f1:.4f} at epoch {best_epoch}', i)

        print(
            '{}Epoch {} validation: loss={} | P={:.4f}, R={:.4f}, F1={:.4f} (TP={}, FP={}, FN={}) | BestF1={:.4f}@{}'.format(
                '[NEW BEST] ' if is_best else '',
                i, val_total_loss, metrics['precision'], metrics['recall'], metrics['f1'],
                total_tp, total_fp, total_fn,
                best_f1, best_epoch
            )
        )
        net.train()



    writer.close()
