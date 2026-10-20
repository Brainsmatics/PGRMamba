"""Infer the full training architecture and evaluate with physical 6 um distances."""
import argparse
import json
from pathlib import Path

import numpy as np

from load_data import dataset0
from inference import load_detector, predict_image
from evaluate_prf1_csv_um import PhysicalDistanceEvaluator, aggregate, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--data_path', type=Path, default=Path(__file__).resolve().parent / '3D_NSD/val')
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--distance_um', type=float, default=6.)
    parser.add_argument('--spacing_xyz', type=float, nargs=3, default=(1., 1., 2.))
    args = parser.parse_args()
    evaluator = PhysicalDistanceEvaluator(args.distance_um, args.spacing_xyz)
    dataset = dataset0(str(args.data_path))
    model = load_detector(args.model_path, args.seed, args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    pred_dir = args.out_dir / 'predictions_csv'
    pred_dir.mkdir()
    rows = []
    for index in range(len(dataset)):
        image, labels = dataset[index]
        predictions = predict_image(model, image, args.distance_um, args.spacing_xyz)
        gt = np.asarray(labels, dtype=np.float32).reshape(-1, 4)[:, :3]
        result = evaluator.match_predictions(predictions[:, :3], gt)
        stem = Path(dataset.image_list[index]).stem
        np.savetxt(pred_dir / f'{stem}.csv', predictions[:, :3], delimiter=',', fmt='%.8g')
        row = dict(block=stem, brain=stem.split('_')[0], gt_count=len(gt), pred_count=len(predictions),
                   tp=result.tp, fp=result.fp, fn=result.fn,
                   **evaluator.compute_metrics(result.tp, result.fp, result.fn))
        rows.append(row)
        print(f"{stem}: P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}")
    overall = aggregate(rows)
    by_brain = {brain: aggregate([r for r in rows if r['brain'] == brain])
                for brain in sorted({r['brain'] for r in rows})}
    write_csv(args.out_dir / 'per_block.csv', rows, list(rows[0]))
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    parameters.update(confidence_threshold=0.5, coordinate_units='voxel',
                      matching='distance-sorted greedy one-to-one, inclusive radius')
    (args.out_dir / 'report.json').write_text(json.dumps(dict(parameters=parameters, overall=overall, by_brain=by_brain), indent=2) + '\n')
    print(json.dumps(overall, indent=2))


if __name__ == '__main__':
    main()
