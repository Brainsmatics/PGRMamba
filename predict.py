"""Predict neuron centers from TIFF volumes without requiring CSV annotations."""
import argparse
import json
from pathlib import Path

import numpy as np

from inference import load_detector, predict_image
from load_data import dataset0
from evaluate_prf1_csv_um import PhysicalDistanceEvaluator, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--data_path', type=Path, required=True, help='Directory of TIFF volumes; CSVs are not needed')
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--distance_um', type=float, default=6.)
    parser.add_argument('--spacing_xyz', type=float, nargs=3, default=(1., 1., 2.))
    args = parser.parse_args()
    PhysicalDistanceEvaluator(args.distance_um, args.spacing_xyz)
    dataset = dataset0(str(args.data_path), require_labels=False)
    model = load_detector(args.model_path, args.seed, args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    pred_dir = args.out_dir / 'predictions_csv'
    pred_dir.mkdir()
    rows = []
    for index in range(len(dataset)):
        image, _ = dataset[index]
        predictions = predict_image(model, image, args.distance_um, args.spacing_xyz)
        stem = Path(dataset.image_list[index]).stem
        np.savetxt(pred_dir / f'{stem}.csv', predictions[:, :3], delimiter=',', fmt='%.8g')
        rows.append(dict(block=stem, pred_count=len(predictions)))
        print(f'{stem}: {len(predictions)} centers')
    write_csv(args.out_dir / 'counts.csv', rows, ['block', 'pred_count'])
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    parameters.update(confidence_threshold=0.5, coordinate_units='voxel', images=len(rows))
    (args.out_dir / 'run_config.json').write_text(json.dumps(parameters, indent=2) + '\n')


if __name__ == '__main__':
    main()
