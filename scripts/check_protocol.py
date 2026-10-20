"""CPU checks for physical thresholds, seed control, data pairing and GCA gradients."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
import numpy as np
import SimpleITK as sitk
import torch
from load_data import dataset0, dataset0_collate
from get_loss import GCA_Loss
from train_detction_um import decode_output, nms_distance_um, set_random_seed
from evaluate_prf1_csv_um import PhysicalDistanceEvaluator


def main():
    evaluator = PhysicalDistanceEvaluator(6., (1., 1., 2.))
    origin = np.zeros((1, 3), dtype=np.float32)
    # A 3-voxel Z separation is exactly 6 um and must match inclusively.
    assert evaluator.match_predictions([[0, 0, 3]], origin).tp == 1
    assert evaluator.match_predictions([[0, 0, 3.01]], origin).tp == 0
    assert evaluator.match_predictions([[6, 0, 0]], origin).tp == 1
    result = evaluator.match_predictions([[0, 0, 0], [0, 0, 1]], origin)
    assert (result.tp, result.fp, result.fn) == (1, 1, 0)
    dets = np.array([[0, 0, 0, .9], [0, 0, 3, .8], [0, 0, 3.01, .7]], dtype=np.float32)
    kept = nms_distance_um(dets)
    np.testing.assert_array_equal(kept, dets[[0, 2]])
    pred = torch.full((4, 2, 2, 2), -100.)
    pred[:3, 1, 0, 1] = 0
    pred[3, 1, 0, 1] = 100
    np.testing.assert_allclose(decode_output(pred), [[2, 6, 6, 1]])
    set_random_seed(2025)
    first = (random.random(), np.random.random(), torch.rand(3))
    set_random_seed(2025)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[:2] == second[:2] and torch.equal(first[2], second[2])
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        volume = np.arange(8 ** 3, dtype=np.float32).reshape(8, 8, 8)
        # Write in reverse name order to verify the stable dataset ordering.
        for name, label in [('B1_002', ''), ('B1_001', '2,3,4\n')]:
            sitk.WriteImage(sitk.GetImageFromArray(volume), str(folder / f'{name}.tif'))
            (folder / f'{name}.csv').write_text(label)
        data = dataset0(str(folder))
        assert [Path(p).stem for p in data.image_list] == ['B1_001', 'B1_002']
        image, label = data[0]
        np.testing.assert_array_equal(image, volume.transpose(0, 2, 1)[None])
        np.testing.assert_array_equal(label, [[2, 3, 4, 1]])
        images, labels = dataset0_collate([data[0], data[1]])
        assert images.shape == (2, 1, 8, 8, 8) and len(labels[1]) == 0
        (folder / 'B1_002.csv').unlink()
        inference_data = dataset0(str(folder), require_labels=False)
        inference_image, inference_labels = inference_data[1]
        np.testing.assert_array_equal(inference_image, volume.transpose(0, 2, 1)[None])
        assert inference_labels.shape == (0, 4)
        try:
            dataset0(str(folder))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing CSV must raise an error')
    for label in (np.array([[2, 3, 4, 1]], dtype=np.float32), np.empty((0, 4), dtype=np.float32)):
        outputs = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
        loss = GCA_Loss(outputs, (label,), torch.device('cpu'), 1.5, 6).get_loss()[0]
        assert torch.isfinite(loss)
        loss.backward()
        assert outputs.grad is not None and torch.isfinite(outputs.grad).all()
    root = Path(__file__).resolve().parents[1] / '3D_NSD'
    if root.is_dir():
        for split in ('train', 'val'):
            images = {p.stem for p in (root / split).glob('*.tif')}
            labels = {p.stem for p in (root / split).glob('*.csv')}
            assert images and images == labels, f'Invalid pairs in {split}'
            print(f'{split}: {len(images)} paired images/CSVs')
    else:
        print('Dataset distributed separately: real-dataset pairing check skipped.')
    print('PASS: 6 um matching/NMS boundaries, anisotropic spacing, one-to-one matching, decoder,')
    print('seed control, TIFF preprocessing, label-free inference, variable-length labels and GCA forward/backward.')


if __name__ == '__main__':
    main()
