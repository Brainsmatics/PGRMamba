"""Standalone checkpoint inference and optional physical-distance PRF1 evaluation."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from evaluate_prf1_csv_um import PhysicalDistanceEvaluator, aggregate, write_csv
from inference import load_detector
from load_data import dataset0
from train_detction_um import decode_output, nms_distance_um


def evaluate(model_path, data_path, out_dir=None, *, device='cuda:0', seed=2025,
             conf_thres=0.5, dist_thres_um=6., nms_dist_thres_um=6.,
             spacing_xyz=(1., 1., 2.), no_gt=False, max_blocks=None, export_swc=False):
    model_path, data_path = Path(model_path), Path(data_path)
    out_dir = Path(out_dir) if out_dir is not None else None
    evaluator = PhysicalDistanceEvaluator(dist_thres_um, spacing_xyz)
    PhysicalDistanceEvaluator(nms_dist_thres_um, spacing_xyz)
    if not np.isfinite(conf_thres) or not 0 <= conf_thres <= 1:
        raise ValueError('conf_thres must be finite and between 0 and 1')
    if max_blocks is not None and max_blocks < 1:
        raise ValueError('max_blocks must be positive')
    if export_swc and out_dir is None:
        raise ValueError('--export_swc requires --out_dir')
    if out_dir is not None and out_dir.exists():
        raise FileExistsError(f'Output directory already exists: {out_dir}; choose a new directory')
    dataset = dataset0(str(data_path), require_labels=not no_gt)
    available_blocks = len(dataset)
    if max_blocks is not None:
        dataset.image_list = dataset.image_list[:max_blocks]
        dataset.label_list = dataset.label_list[:max_blocks]
    model = load_detector(model_path, seed=seed, device=device)
    kind = {6: '2D-no-div', 7: '2D', 8: '3D'}.get(model.mamba.ehance.av.in_channels)
    print(f'{kind}-PGE checkpoint loaded strictly: {model_path}', flush=True)
    csv_dir = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=False)
        csv_dir = out_dir / 'predictions_csv'
        csv_dir.mkdir()
    if export_swc:
        (out_dir / 'predictions_swc_um').mkdir()
    rows, matches = [], []
    with torch.no_grad():
        for index in range(len(dataset)):
            image, labels = dataset[index]
            stem = Path(dataset.image_list[index]).stem
            if image.ndim != 4 or image.shape[0] != 1 or not np.isfinite(image).all():
                raise ValueError(f'Invalid TIFF input: {stem}, shape={image.shape}')
            if any(size % 4 for size in image.shape[1:]):
                raise ValueError(f'All TIFF axis sizes must be divisible by 4: {stem}, shape={image.shape}')
            output = model(torch.as_tensor(image[None], device=device))
            raw = decode_output(output[0], conf_thres=conf_thres)
            predictions = nms_distance_um(raw, nms_dist_thres_um, spacing_xyz)
            if csv_dir is not None:
                np.savetxt(csv_dir / f'{stem}.csv', predictions[:, :3], delimiter=',', fmt='%.9g')
            row = dict(block=stem, brain=stem.split('_')[0], pred_count=len(predictions))
            if not no_gt:
                gt = np.asarray(labels, dtype=np.float32).reshape(-1, 4)[:, :3]
                extent = np.array([image.shape[2], image.shape[3], image.shape[1]])
                if not np.isfinite(gt).all() or (gt < 0).any() or (gt >= extent).any():
                    raise ValueError(f'Invalid or out-of-bounds GT coordinates: {stem}')
                result = evaluator.match_predictions(predictions[:, :3], gt)
                density = 'sparse' if len(gt) < 80 else ('medium' if len(gt) < 300 else 'dense')
                row.update(gt_count=len(gt), density=density, tp=result.tp, fp=result.fp, fn=result.fn,
                           **evaluator.compute_metrics(result.tp, result.fp, result.fn))
                matches.extend(dict(block=stem, pred_index=pi, gt_index=gi, distance_um=distance)
                               for pi, gi, distance in result.matches)
                print(f"[{index + 1}/{len(dataset)}] {stem}: TP={result.tp} FP={result.fp} FN={result.fn} "
                      f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}", flush=True)
            else:
                print(f'[{index + 1}/{len(dataset)}] {stem}: {len(predictions)} centers', flush=True)
            if export_swc:
                xyz_um = predictions[:, :3].astype(np.float64) * evaluator.spacing_xyz
                with (out_dir / 'predictions_swc_um' / f'{stem}.swc').open('w') as handle:
                    for number, (x, y, z) in enumerate(xyz_um, 1):
                        handle.write(f'{number} 1 {x:.6f} {y:.6f} {z:.6f} 1 -1\n')
            rows.append(row)
    parameters = dict(model_path=str(model_path.resolve()), data_path=str(data_path.resolve()),
                      pge=kind, strict_weight_load=True, device=device, seed=seed,
                      confidence_threshold=conf_thres, matching_radius_um=dist_thres_um,
                      nms_radius_um=nms_dist_thres_um, spacing_xyz_um=list(spacing_xyz),
                      prediction_csv_units='voxel', swc_units='um' if export_swc else None,
                      evaluated_blocks=len(dataset), available_blocks=available_blocks, no_gt=no_gt)
    report = dict(parameters=parameters, per_block=rows)
    if not no_gt:
        report.update(overall=aggregate(rows),
                      by_brain={brain: aggregate([r for r in rows if r['brain'] == brain])
                                for brain in sorted({r['brain'] for r in rows})},
                      by_density={d: aggregate([r for r in rows if r['density'] == d])
                                  for d in ('sparse', 'medium', 'dense')})
        if out_dir is not None:
            brain_rows = [dict(brain=brain, **values) for brain, values in report['by_brain'].items()]
            write_csv(out_dir / 'per_brain_metrics.csv', brain_rows, list(brain_rows[0]))
            write_csv(out_dir / 'matches_um.csv', matches, ['block', 'pred_index', 'gt_index', 'distance_um'])
        print(json.dumps(report['overall'], indent=2), flush=True)
    if out_dir is not None:
        write_csv(out_dir / ('counts.csv' if no_gt else 'per_block_metrics.csv'), rows, list(rows[0]))
        (out_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', type=Path, required=True, help='Path to a compatible model checkpoint')
    parser.add_argument('--data_path', type=Path, required=True, help='Directory of TIFF blocks')
    parser.add_argument('--out_dir', type=Path, default=None, help='Optional new output directory; omit to print results without saving files')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--conf_thres', type=float, default=0.5)
    parser.add_argument('--dist_thres_um', type=float, default=6.)
    parser.add_argument('--nms_dist_thres_um', type=float, default=6.)
    parser.add_argument('--spacing_xyz', type=float, nargs=3, default=(1., 1., 2.))
    parser.add_argument('--no_gt', action='store_true', help='Predict without reading CSV labels or computing metrics')
    parser.add_argument('--max_blocks', type=int, default=None)
    parser.add_argument('--export_swc', action='store_true', help='Also export centers as SWC in micrometers')
    evaluate(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
