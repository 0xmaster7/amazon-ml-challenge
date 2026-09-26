"""
error_analysis.py — Bucket your validation errors before touching the model.

Strategy doc point 1: false merges (cost 2x), false negatives by noise type,
recall-ceiling failures (true match never blocked). Run after train.py:

    python error_analysis.py --data_dir <train dir> --output_dir <out dir>

Reads the checkpoints train.py saved (features_train.pkl, oof_proba.npy,
xgb_model.pkl) and the ground truth. No recomputation.
"""
import pandas as pd
import numpy as np
import os
import argparse
import pickle


def build_true_dict(gt):
    y_true = {}
    for _, row in gt.iterrows():
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or str(matched).strip() == '':
            y_true[row['source1_entity_id']] = set()
        else:
            y_true[row['source1_entity_id']] = set(str(matched).replace(' ', '').split(','))
    return y_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../../output")
    args = parser.parse_args()

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    df_features = pickle.load(open(os.path.join(ckpt_dir, "features_train.pkl"), 'rb'))
    oof_proba = np.load(os.path.join(ckpt_dir, "oof_proba.npy"))
    saved = pickle.load(open(os.path.join(args.output_dir, "xgb_model.pkl"), 'rb'))
    thresholds = saved.get('thresholds', {'default': 0.5})
    gt = pd.read_csv(os.path.join(args.data_dir, "train_ground_truth.tsv"), sep="\t")
    y_true = build_true_dict(gt)

    df = df_features.copy()
    df['proba'] = oof_proba
    df['pred'] = (df['proba'] > thresholds['default']).astype(int)

    print("=" * 60)
    print("ERROR ANALYSIS (out-of-fold, global threshold "
          f"{thresholds['default']:.2f})")
    print("=" * 60)

    # ---------- 1. FALSE MERGES (predicted match, not true) ----------
    fp = df[(df['pred'] == 1) & (df['is_true_match'] == 0)]
    print(f"\n1. FALSE MERGES: {len(fp)} pairs (each one costs 2x under F_0.5)")
    if len(fp) > 0:
        print("   Profile of false merges vs true matches (mean feature values):")
        tp = df[df['is_true_match'] == 1]
        for col in ['name_jaro_winkler', 'name_token_set_ratio', 'embedding_cosine_sim',
                    'cross_encoder_score', 'addr_fuzzy_ratio', 'pincode_exact_match',
                    'country_match', 'address_missing', 'total_layers_caught']:
            if col in df.columns:
                print(f"     {col:28s} FP={fp[col].mean():.3f}  TP={tp[col].mean():.3f}")
        # Chain/franchise trap: high name sim, different address
        chains = fp[(fp['name_token_set_ratio'] > 0.9) & (fp['addr_fuzzy_ratio'] < 0.4)]
        print(f"   Classic chain/franchise trap (name near-identical, address different): "
              f"{len(chains)} ({100*len(chains)/max(len(fp),1):.1f}% of false merges)")
        # False merges that hit true singletons (drop 1.0 -> 0.0)
        singleton_hits = fp[fp['source1_entity_id'].isin([k for k, v in y_true.items() if not v])]
        print(f"   False merges landing on TRUE SINGLETONS (entity score 1.0 -> 0.0): "
              f"{singleton_hits['source1_entity_id'].nunique()} entities")

    # ---------- 2. FALSE NEGATIVES (true match, predicted no) ----------
    fn = df[(df['pred'] == 0) & (df['is_true_match'] == 1)]
    print(f"\n2. FALSE NEGATIVES: {len(fn)} true pairs missed at threshold")
    if len(fn) > 0:
        for label, mask in [
            ("mojibake on either side", (fn.get('is_mojibake_s1', 0) == 1) | (fn.get('is_mojibake_cand', 0) == 1)),
            ("address missing", fn.get('address_missing', 0) == 1),
            ("caught by ONLY ONE blocking layer", fn.get('total_layers_caught', 2) == 1),
        ]:
            try:
                n = int(mask.sum())
                print(f"     {label}: {n} ({100*n/len(fn):.1f}%)")
            except Exception:
                pass
        print("   Probability distribution of missed true pairs "
              "(if these cluster near a fixed band, widen the LLM band there):")
        hist = np.histogram(fn['proba'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[0]
        for lo, hi, c in zip([0, .1, .2, .3, .4, .5, .6, .7, .8, .9], [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], hist):
            if c:
                print(f"     {lo:.1f}-{hi:.1f}: {c}")

    # ---------- 3. RECALL-CEILING FAILURES ----------
    print("\n3. RECALL CEILING (true pairs blocking never retrieved):")
    total_true = int(df['is_true_match'].sum())
    print(f"   True pairs inside candidate set: {total_true}")
    gt_pairs = set()
    for s1, matches in y_true.items():
        for m in matches:
            gt_pairs.add((s1, m))
    blocked = set(zip(df['source1_entity_id'], df['candidate_entity_id']))
    ceiling_misses = gt_pairs - blocked
    print(f"   True pairs in GT total: {len(gt_pairs)}")
    print(f"   Never blocked (unrecoverable at threshold): {len(ceiling_misses)} "
          f"({100*len(ceiling_misses)/max(len(gt_pairs),1):.1f}%)")
    if ceiling_misses:
        print(f"   -> Your score is hard-capped below {1 - len(ceiling_misses)/max(len(gt_pairs),1):.3f} "
              f"on recall alone. Raise FAISS top_k, lower its threshold, or add --embedder2.")

    # ---------- 4. LLM BAND SUGGESTION ----------
    print("\n4. LLM BAND: where the classifier is actually wrong")
    err_probas = pd.concat([fp['proba'], fn['proba']])
    if len(err_probas) > 0:
        q = np.percentile(err_probas, [10, 90])
        print(f"   80% of errors sit in proba range [{q[0]:.2f}, {q[1]:.2f}].")
        print(f"   Current band covers threshold +/- 0.15. If the error range is wider, "
              f"re-run with --band_lo {q[0]:.2f} --band_hi {q[1]:.2f}.")

    print("\nDone. Fix order: ceiling misses (blocking) -> false merges on singletons "
          "-> the biggest FN noise bucket -> band tuning.")


if __name__ == "__main__":
    main()
