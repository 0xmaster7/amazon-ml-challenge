"""
scorer.py — the competition metric, extracted so scripts that only need scoring
(smoke tests, error analysis) don't have to import xgboost.

Macro F_0.5 over S1 entities. Empty GT + empty prediction = 1.0;
empty GT + any prediction = 0.0. beta=0.5 weighs precision twice as heavily
as recall. From the official evaluation_methodology.md spec.
"""
import numpy as np
import pandas as pd

def f_05_per_entity(y_true_dict, y_pred_dict):
    scores = []
    for s1_id in y_true_dict:
        true_set = y_true_dict[s1_id]
        pred_set = y_pred_dict.get(s1_id, set())
        if len(true_set) == 0 and len(pred_set) == 0:
            scores.append(1.0)
        elif len(true_set) == 0 and len(pred_set) > 0:
            scores.append(0.0)
        elif len(true_set) > 0 and len(pred_set) == 0:
            scores.append(0.0)
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                scores.append(0.0)
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                scores.append(f05)
    return np.mean(scores) if scores else 0.0




def build_true_dict(gt):
    y_true = {}
    for _, row in gt.iterrows():
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or str(matched).strip() == '':
            y_true[row['source1_entity_id']] = set()
        else:
            y_true[row['source1_entity_id']] = set(str(matched).replace(' ', '').split(','))
    return y_true


def macro_score_from_proba(y_true_dict, s1_arr, cand_arr, proba, thresh):
    y_pred = {k: set() for k in y_true_dict}
    mask = proba > thresh
    for s1, c in zip(s1_arr[mask], cand_arr[mask]):
        if s1 in y_pred:
            y_pred[s1].add(c)
    return f_05_per_entity(y_true_dict, y_pred)


# ============================================================
# CHECKPOINTING (session-cutoff insurance)
# Delete output/checkpoints/ if you change cleaning, blocking, or
# feature code - stale checkpoints are not invalidated automatically.
# ============================================================
