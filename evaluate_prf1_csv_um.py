"""Evaluate voxel-coordinate CSVs using a physical distance threshold in micrometers.

Defaults: radius = 6 um; voxel spacing (x, y, z) = (1, 1, 2) um.
Uses the original evaluator's distance-sorted greedy one-to-one matching and PRF1.
Only the distance calculation changes: d = norm((pred_xyz - gt_xyz) * spacing_xyz).
No confidence thresholding, NMS, coordinate rounding, or resampling is applied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import SWCEvaluator


def _write_swc(points: np.ndarray, out_path: Path, radius: float = 1.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for i, (x, y, z) in enumerate(points, start=1):
            f.write(f'{i} 1 {x} {y} {z} {radius} -1\n')


class PhysicalDistanceEvaluator(SWCEvaluator):
    """Input points are x/y/z voxel coordinates; returned match distances are um."""

    def __init__(self, distance_threshold_um=6.0, spacing_xyz=(1.0, 1.0, 2.0)):
        threshold = float(distance_threshold_um)
        spacing = np.asarray(spacing_xyz, dtype=np.float64)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('distance_threshold_um must be finite and positive')
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError('spacing_xyz must contain three finite positive um/voxel values')
        super().__init__(distance_threshold=threshold)
        self.spacing_xyz = spacing

    def _distance_matrix(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        # Preserve the original float32 input coordinates; perform physical geometry in float64.
        diff_um = (pred.astype(np.float64)[:, None, :] - gt.astype(np.float64)[None, :, :])
        diff_um *= self.spacing_xyz
        return np.sqrt(np.sum(diff_um * diff_um, axis=2))


def load_csv_coords(path: Path) -> np.ndarray:
    """Read headerless x,y,z[,extra columns]; reject missing or malformed files.

    An existing empty CSV is valid and represents zero detections/labels.
    """
    if not path.is_file():
        raise FileNotFoundError(f'CSV missing: {path}. For zero predictions, supply an empty CSV.')
    if not path.read_text().strip():
        return np.empty((0, 3), dtype=np.float32)
    try:
        data = np.loadtxt(path, delimiter=',', dtype=np.float32, ndmin=2)
    except ValueError as exc:
        raise ValueError(f'Invalid CSV {path}: expected headerless numeric x,y,z columns') from exc
    if data.shape[1] < 3 or not np.isfinite(data[:, :3]).all():
        raise ValueError(f'Invalid coordinates in {path}: need at least 3 finite columns')
    return data[:, :3]


def resolve_prediction(pred_dir: Path, gt_path: Path) -> Path:
    # Same filename precedence as evaluate_prf1_csv.py.
    candidates = [pred_dir / f'{gt_path.stem}_3D.csv',
                  pred_dir / (gt_path.stem.replace('_z', '_masks_z') + '.tif.csv'),
                  pred_dir / gt_path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'No prediction CSV for {gt_path.name} in {pred_dir}')


def aggregate(rows: list[dict]) -> dict:
    counts = {key: sum(r[key] for r in rows) for key in ('tp', 'fp', 'fn')}
    return dict(blocks=len(rows), **counts, **SWCEvaluator.compute_metrics(**counts))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(pred_dir: Path, gt_dir: Path, dist_thres_um: float = 6.0,
             spacing_xyz=(1.0, 1.0, 2.0), swc_dir: Path | None = None,
             out_dir: Path | None = None) -> dict:
    pred_dir, gt_dir = Path(pred_dir).resolve(), Path(gt_dir).resolve()
    evaluator = PhysicalDistanceEvaluator(dist_thres_um, spacing_xyz)
    gt_files = sorted(gt_dir.glob('*.csv'))
    if not gt_files:
        raise FileNotFoundError(f'No GT CSVs in {gt_dir}')
    print(f'Physical matching: distance <= {dist_thres_um:g} um; spacing x/y/z = '
          f'{tuple(evaluator.spacing_xyz.tolist())} um/voxel; input coordinates = x,y,z voxels')
    rows, matches, inputs = [], [], []
    for index, gt_path in enumerate(gt_files, 1):
        pred_path = resolve_prediction(pred_dir, gt_path)
        gt, pred = load_csv_coords(gt_path), load_csv_coords(pred_path)
        result = evaluator.match_predictions(pred, gt)
        assert result.tp + result.fp == len(pred) and result.tp + result.fn == len(gt)
        density = 'sparse' if len(gt) < 80 else ('medium' if len(gt) < 300 else 'dense')
        row = dict(block=gt_path.stem, brain=gt_path.stem.split('_')[0], density=density,
                   gt_count=len(gt), pred_count=len(pred), tp=result.tp, fp=result.fp, fn=result.fn,
                   **evaluator.compute_metrics(result.tp, result.fp, result.fn))
        rows.append(row)
        matches.extend(dict(block=gt_path.stem, pred_index=pi, gt_index=gi, distance_um=d)
                       for pi, gi, d in result.matches)
        print(f"Image {index:04d}: {gt_path.name} | TP={result.tp:4d} FP={result.fp:4d} FN={result.fn:4d} | "
              f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}")
        if swc_dir is not None:
            _write_swc(pred, Path(swc_dir) / (gt_path.stem + '.swc'))
        for path in (gt_path, pred_path):
            inputs.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    overall = aggregate(rows)
    by_density = {d: aggregate([r for r in rows if r['density'] == d]) for d in ('sparse', 'medium', 'dense')}
    by_brain = {b: aggregate([r for r in rows if r['brain'] == b]) for b in sorted({r['brain'] for r in rows})}
    print('\nDensity-based evaluation (GT counts: sparse <80, medium 80–299, dense >=300)')
    for density, group in by_density.items():
        print(f"{density}: blocks={group['blocks']} TP={group['tp']} FP={group['fp']} FN={group['fn']} | "
              f"P={group['precision']:.4f} R={group['recall']:.4f} F1={group['f1']:.4f}")
    print('\nBrain-level evaluation (micro aggregation across blocks)')
    for brain, group in by_brain.items():
        print(f"{brain}: blocks={group['blocks']} TP={group['tp']} FP={group['fp']} FN={group['fn']} | "
              f"P={group['precision']:.4f} R={group['recall']:.4f} F1={group['f1']:.4f}")
    print(f"\nOverall: blocks={overall['blocks']} TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} | "
          f"P={overall['precision'] * 100:.2f} R={overall['recall'] * 100:.2f} F1={overall['f1'] * 100:.2f} (percent)")
    report = dict(parameters=dict(distance_threshold_um=float(dist_thres_um),
        spacing_xyz_um=evaluator.spacing_xyz.tolist(), input_coordinate_units='voxel',
        match_distance_units='um', matching='distance-sorted greedy one-to-one; inclusive threshold',
        aggregation='sum TP/FP/FN, then compute precision/recall/F1',
        pred_dir=str(pred_dir), gt_dir=str(gt_dir), swc_coordinate_units='voxel'),
        overall=overall, by_density=by_density, by_brain=by_brain, per_block=rows)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'summary.json').write_text(json.dumps(report, indent=2))
        write_csv(out_dir / 'per_block_metrics.csv', rows, list(rows[0]))
        write_csv(out_dir / 'matches_um.csv', matches, ['block', 'pred_index', 'gt_index', 'distance_um'])
        brain_rows = [dict(brain=b, **v) for b, v in by_brain.items()]
        write_csv(out_dir / 'per_brain_metrics.csv', brain_rows, list(brain_rows[0]))
        code_paths = [Path(__file__).resolve(), PROJECT / 'evaluation.py', PROJECT / 'evaluate_prf1_csv.py']
        sources = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in code_paths]
        (out_dir / 'provenance.json').write_text(json.dumps(dict(inputs=inputs, sources=sources), indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred_dir', required=True, type=Path)
    parser.add_argument('--gt_dir', required=True, type=Path)
    parser.add_argument('--dist_thres_um', type=float, default=6.0, help='Physical matching radius in um (default: 6)')
    parser.add_argument('--spacing_xyz', nargs=3, type=float, default=(1.0, 1.0, 2.0),
                        metavar=('SX', 'SY', 'SZ'), help='Voxel spacing in x,y,z order, um/voxel (default: 1 1 2)')
    parser.add_argument('--swc_dir', type=Path, default=None, help='Optional SWC export; retain original voxel coordinates')
    parser.add_argument('--out_dir', type=Path, default=None, help='Optional directory for metrics, match distances and provenance')
    args = parser.parse_args()
    evaluate(args.pred_dir, args.gt_dir, args.dist_thres_um, args.spacing_xyz, args.swc_dir, args.out_dir)


if __name__ == '__main__':
    main()
