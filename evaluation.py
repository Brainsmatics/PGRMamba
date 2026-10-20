from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    matches: list[tuple[int, int, float]]  # (pred_idx, gt_idx, distance)


class SWCEvaluator:
    """Simple point-based evaluator using distance-threshold 1-to-1 matching.

    - Each prediction can match at most one GT.
    - Each GT can match at most one prediction.
    - A match is valid if distance <= distance_threshold.

    Matching is done greedily by sorting all candidate pairs by distance.
    """

    def __init__(self, distance_threshold: float = 10.0):
        self.distance_threshold = float(distance_threshold)

    @staticmethod
    def _distance_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        # pred: (P, 3), gt: (G, 3)
        # returns: (P, G)
        diff = pred[:, None, :] - gt[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float64))
        return dist

    def match_predictions(self, pred_coords: np.ndarray, gt_coords: np.ndarray) -> MatchResult:
        pred_coords = np.asarray(pred_coords, dtype=np.float32)
        gt_coords = np.asarray(gt_coords, dtype=np.float32)

        if pred_coords.size == 0 and gt_coords.size == 0:
            return MatchResult(tp=0, fp=0, fn=0, matches=[])
        if pred_coords.size == 0:
            return MatchResult(tp=0, fp=0, fn=int(gt_coords.shape[0]), matches=[])
        if gt_coords.size == 0:
            return MatchResult(tp=0, fp=int(pred_coords.shape[0]), fn=0, matches=[])

        P = pred_coords.shape[0]
        G = gt_coords.shape[0]
        dist = self._distance_matrix(pred_coords, gt_coords)

        # candidate pairs within threshold
        cand = np.argwhere(dist <= self.distance_threshold)
        if cand.size == 0:
            return MatchResult(tp=0, fp=int(P), fn=int(G), matches=[])

        cand_dist = dist[cand[:, 0], cand[:, 1]]
        order = np.argsort(cand_dist)
        cand = cand[order]
        cand_dist = cand_dist[order]

        pred_used = np.zeros((P,), dtype=bool)
        gt_used = np.zeros((G,), dtype=bool)
        matches: list[tuple[int, int, float]] = []

        for (p_idx, g_idx), d in zip(cand, cand_dist):
            if pred_used[p_idx] or gt_used[g_idx]:
                continue
            pred_used[p_idx] = True
            gt_used[g_idx] = True
            matches.append((int(p_idx), int(g_idx), float(d)))

        tp = int(np.sum(pred_used))
        fp = int(P - tp)
        fn = int(G - int(np.sum(gt_used)))
        return MatchResult(tp=tp, fp=fp, fn=fn, matches=matches)

    @staticmethod
    def compute_metrics(tp: int, fp: int, fn: int) -> dict:
        tp = int(tp)
        fp = int(fp)
        fn = int(fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
