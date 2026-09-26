import pandas as pd
import numpy as np
import os
import argparse
import pickle
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from data_cleaner import process_dataframe
from blocker import LayeredBlocker
from feature_engineering import build_features_for_pairs
from scorer import build_true_dict, macro_score_from_proba
from llm_sniper import LLMSniper


def save_ckpt(ckpt_dir, name, obj):
    path = os.path.join(ckpt_dir, name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"[checkpoint] saved {name} ({os.path.getsize(path)/1e6:.1f} MB)")

def load_ckpt(ckpt_dir, name):
    path = os.path.join(ckpt_dir, name)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"[checkpoint] resumed from {name}")
        return obj
    return None


# ============================================================
# F-0.5 SCORER — PER-ENTITY MACRO AVERAGE (Competition Formula)
# Edge cases verified against the spec (see smoke_tests.py):
#   empty GT + empty pred = 1.0 ; empty GT + any pred = 0.0 ;
#   non-empty GT + empty pred = 0.0 (no division error, no skip) ;
#   macro per-entity average, never micro/pooled.
# ============================================================
# ============================================================
# XGBOOST TRAINING: seed-ensembled folds, OOF calibration,
# optional per-country stratified thresholds
# ============================================================
def train_xgboost(df_features, labels, groups, strata, output_dir, gt, seeds, stratify, no_spw):
    print("\n========== TRAINING XGBOOST ==========")
    exclude = ['source1_entity_id', 'candidate_entity_id', 'is_true_match', 'match_label']
    feature_cols = [c for c in df_features.columns if c not in exclude
                    and df_features[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]

    print(f"Using {len(feature_cols)} features; seeds {seeds}; stratified thresholds: {stratify}")
    X = df_features[feature_cols]
    y = labels.values if hasattr(labels, 'values') else labels

    y_true_dict = build_true_dict(gt)
    s1_arr = df_features['source1_entity_id'].values
    cand_arr = df_features['candidate_entity_id'].values

    # Entity-grouped holdout (no entity straddles train/val), stratified by
    # the entity's country so every fold carries both India and US entities
    # (strategy doc: noise profiles differ by country).
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    oof_proba = np.zeros(len(X))
    all_fold_scores = []

    # stratify folds by the entity's country (per-row via entity), not the label
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, strata, groups=groups)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Seed ensembling (strategy doc point 4): same fold, different seeds ->
        # average probabilities. colsample_bytree already varies feature
        # subsets per tree; different seeds change those subsets.
        fold_val_proba = np.zeros(len(X_val))
        for seed in seeds:
            spw = 1.0 if no_spw else len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
            clf = xgb.XGBClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=7,
                subsample=0.8, colsample_bytree=0.8,
                early_stopping_rounds=50, random_state=seed,
                tree_method="hist", device="cuda",
                scale_pos_weight=spw
            )
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
            models.append(clf)
            fold_val_proba += clf.predict_proba(X_val)[:, 1]
        fold_val_proba /= len(seeds)
        oof_proba[val_idx] = fold_val_proba

        fold_score = macro_score_from_proba(y_true_dict, s1_arr[val_idx], cand_arr[val_idx], fold_val_proba, 0.5)
        all_fold_scores.append(fold_score)
        print(f"Fold {fold+1} MACRO F-0.5 @0.50: {fold_score:.4f}")

    print(f"\n>>> Average per-fold MACRO F-0.5 @0.50: {np.mean(all_fold_scores):.4f}")

    # ---- Threshold sweep on OOF probabilities ----
    def sweep(mask=None):
        best_t, best_s = 0.5, -1.0
        idx = np.arange(len(X)) if mask is None else np.where(mask)[0]
        for ti in range(30, 98):
            t = ti / 100.0
            s = macro_score_from_proba(y_true_dict, s1_arr[idx], cand_arr[idx], oof_proba[idx], t)
            if s > best_s:
                best_s, best_t = s, t
        return best_t, best_s

    global_thresh, global_score = sweep()
    print(f">>> OOF MACRO F-0.5 (global threshold): {global_score:.4f} at {global_thresh:.2f}")

    thresholds = {'default': global_thresh}
    if stratify:
        # Per-country thresholds (strategy doc point 5): India vs US noise
        # profiles differ; sweep each stratum separately. Countries with too
        # few pairs fall back to the global threshold.
        row_country = np.asarray(strata)
        for country in sorted(set(row_country)):
            mask = row_country == country
            if mask.sum() < 2000:
                print(f"    stratum {country}: only {mask.sum()} pairs - using global threshold")
                continue
            t, s = sweep(mask)
            thresholds[country] = t
            print(f"    stratum {country}: threshold {t:.2f} (OOF {s:.4f}, {mask.sum()} pairs)")
    if global_thresh >= 0.97 or any(t >= 0.97 for t in thresholds.values()):
        print(">>> WARNING: a threshold hit the sweep ceiling - widen the range upward.")

    # Save OOF probabilities for error_analysis.py
    np.save(os.path.join(output_dir, "checkpoints", "oof_proba.npy"), oof_proba)

    model_path = os.path.join(output_dir, "xgb_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'models': models, 'thresholds': thresholds, 'features': feature_cols,
                     'n_seeds': len(seeds)}, f)
    print(f"Models saved to {model_path}")

    return models, thresholds, feature_cols, oof_proba


def proba_to_preds(row_country, proba, thresholds):
    """Row-wise threshold application: per-country stratum if tuned, else global.
    row_country: per-row normalized country of the S1 entity (pass an array -
    the country columns are dropped from the feature frame)."""
    if len(thresholds) == 1:
        return (proba > thresholds['default']).astype(int)
    t = np.array([thresholds.get(c, thresholds['default']) for c in row_country])
    return (proba > t).astype(int)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../../output")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--no_cross_encoder", action="store_true",
                        help="Disable the cross-encoder feature. Must match predict.py.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", type=str, default="42,7,123",
                        help="Comma-separated seeds for per-fold seed ensembling.")
    parser.add_argument("--stratify", action="store_true",
                        help="Tune thresholds per country stratum instead of one global.")
    parser.add_argument("--no_spw", action="store_true",
                        help="Ablation: scale_pos_weight=1 (test against precision-heavy metric).")
    parser.add_argument("--embedder2", type=str, default=None,
                        help="Optional second multilingual embedder; its neighbors are unioned into blocking.")
    parser.add_argument("--band_lo", type=float, default=None, help="LLM band low (default: threshold-0.15)")
    parser.add_argument("--band_hi", type=float, default=None, help="LLM band high (default: threshold+0.15)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    nrows = 1000 if args.test_mode else None

    # =================== LOAD & CLEAN ===================
    print("========== LOADING DATA ==========")
    cleaned = load_ckpt(ckpt_dir, "cleaned_train.pkl") if args.resume else None
    if cleaned is not None:
        df_s1, df_s2, df_s3 = cleaned
    else:
        df_s1 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source1.tsv"), sep="\t", nrows=nrows))
        df_s2 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source2.tsv"), sep="\t", nrows=nrows))
        df_s3 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source3.tsv"), sep="\t", nrows=nrows))
        save_ckpt(ckpt_dir, "cleaned_train.pkl", (df_s1, df_s2, df_s3))
    df_pool = pd.concat([df_s2, df_s3]).reset_index(drop=True)

    gt = pd.read_csv(os.path.join(args.data_dir, "train_ground_truth.tsv"), sep="\t")

    # =================== SINGLETON + FAN-OUT + CONFLICT ASSUMPTION ===========
    print("\n========== GROUND TRUTH PROFILE ==========")
    singleton_mask = gt['matched_entity_ids'].isna() | (gt['matched_entity_ids'].astype(str).str.strip() == "")
    n_singletons = singleton_mask.sum()
    n_total = len(gt)
    print(f"Singletons (no true match): {n_singletons}/{n_total} ({100*n_singletons/n_total:.1f}%) "
          f"- each false merge on one of these drops it from 1.0 to 0.0")

    gt_non_empty = gt[~singleton_mask].copy()
    gt_non_empty['matched_entity_ids'] = gt_non_empty['matched_entity_ids'].astype(str).str.replace(' ', '')
    gt_non_empty['match_count'] = gt_non_empty['matched_entity_ids'].str.split(',').apply(len)
    max_fanout = gt_non_empty['match_count'].max()
    total_gt_pairs = int(gt_non_empty['match_count'].sum())
    print(f"Max fan-out: {max_fanout} (FAISS top_k=30 must stay above this)")
    print(f"Total GT pairs: {total_gt_pairs}")

    # Verify the conflict-resolution assumption: does any S2/S3 id appear
    # under two different S1 entities in ground truth?
    exploded_ids = gt_non_empty['matched_entity_ids'].str.split(',').explode()
    dup = exploded_ids[exploded_ids.duplicated()]
    if len(dup) > 0:
        print(f"WARNING: {dup.nunique()} pool IDs are claimed by multiple S1 entities in GROUND TRUTH "
              f"- post-processing conflict resolution in predict.py would delete TRUE matches. "
              f"Run predict with --no_conflict_resolution.")
    else:
        print("Conflict check: no pool ID is shared across S1 entities in GT - "
              "conflict resolution in predict.py is safe.")

    # =================== BLOCKING ===================
    print("\n========== STAGE 1: LAYERED BLOCKING ==========")
    blocker_keep = None
    df_pairs = load_ckpt(ckpt_dir, "pairs_train.pkl") if args.resume else None
    if df_pairs is None:
        blocker_keep = LayeredBlocker()
        # Per-layer checkpointing: each completed layer survives an OOM/kill.
        done_layers = blocker_keep.load_progress(ckpt_dir) if args.resume else set()

        def run_layer(name, fn):
            if name in done_layers:
                print(f"[checkpoint] skipping {name} (already done)")
                return
            fn()
            done_layers.add(name)
            blocker_keep.save_progress(ckpt_dir, done_layers)

        run_layer('layer1', lambda: blocker_keep.layer1_exact_key_blocking(df_s1, df_s2, df_s3))
        run_layer('layer2', lambda: blocker_keep.layer2_tfidf_blocking(df_s1, df_s2, df_s3))
        # Layer 3 ALWAYS runs: its memmap cache skips re-encoding, and the
        # feature stage needs the embeddings + id maps attached. Pairs it adds
        # are idempotent, and we save progress right after so a later-layer
        # kill doesn't lose them.
        blocker_keep.layer3_semantic_embeddings(df_s1, df_s2, df_s3)
        if args.embedder2:
            blocker_keep.layer3_semantic_embeddings(df_s1, df_s2, df_s3, model_name=args.embedder2)
        blocker_keep.save_progress(ckpt_dir, done_layers)
        run_layer('layer4', lambda: blocker_keep.layer4_address_only(df_s1, df_s2, df_s3))
        run_layer('layer4b', lambda: blocker_keep.layer4b_near_exact_address(df_s1, df_s2, df_s3))
        run_layer('layer5', lambda: blocker_keep.layer5_phone_key_blocking(df_s1, df_s2, df_s3))
        df_pairs = blocker_keep.export_candidate_pairs(os.path.join(args.output_dir, "candidate_pairs.tsv"))
        save_ckpt(ckpt_dir, "pairs_train.pkl", df_pairs)

    # =================== GROUND TRUTH LABELING ===================
    print("\n========== LABELING CANDIDATES ==========")
    gt_exploded = gt_non_empty.assign(
        matched_entity_ids=gt_non_empty['matched_entity_ids'].str.split(',')
    ).explode('matched_entity_ids')
    gt_exploded['matched_entity_ids'] = gt_exploded['matched_entity_ids'].str.strip()
    gt_exploded['is_true_match'] = 1

    df_pairs = df_pairs.merge(
        gt_exploded[['source1_entity_id', 'matched_entity_ids', 'is_true_match']],
        left_on=['source1_entity_id', 'candidate_entity_id'],
        right_on=['source1_entity_id', 'matched_entity_ids'],
        how='left'
    )
    df_pairs['is_true_match'] = df_pairs['is_true_match'].fillna(0).astype(int)
    if 'matched_entity_ids' in df_pairs.columns:
        df_pairs.drop('matched_entity_ids', axis=1, inplace=True)

    print(f"Positive pairs: {df_pairs['is_true_match'].sum()} | Negative pairs: {(df_pairs['is_true_match'] == 0).sum()}")

    # =================== BLOCKING RECALL ===================
    print("\n========== BLOCKING RECALL (the ceiling) ==========")
    found = int(df_pairs['is_true_match'].sum())
    print(f"Overall: {found}/{total_gt_pairs} = {found/max(total_gt_pairs,1):.4f}")
    layer_cols = [('found_in_layer1', 'L1 exact-key'), ('found_in_layer2', 'L2 tfidf'),
                  ('found_in_layer3', 'L3 faiss'), ('found_in_layer4', 'L4 address-exact'),
                  ('found_in_layer4b', 'L4b address-near'), ('found_in_layer5', 'L5 phone-key')]
    for col, name in layer_cols:
        if col in df_pairs.columns:
            f = int(df_pairs.loc[df_pairs[col] == 1, 'is_true_match'].sum())
            print(f"  {name}: {f}/{total_gt_pairs} ({f/max(total_gt_pairs,1):.4f})")
    gt_counts = gt_non_empty.set_index('source1_entity_id')['match_count']
    found_counts = df_pairs[df_pairs['is_true_match'] == 1].groupby('source1_entity_id').size()
    per_entity_recall = (found_counts.reindex(gt_counts.index).fillna(0) / gt_counts)
    print(f"Per-entity: mean {per_entity_recall.mean():.4f}; fully covered {(per_entity_recall == 1.0).mean():.4f}; "
          f"ZERO candidates {(per_entity_recall == 0.0).mean():.4f}")
    print("Every recall point lost here is unrecoverable downstream.")

    # =================== FEATURE ENGINEERING ===================
    print("\n========== STAGE 2: FEATURE ENGINEERING ==========")
    df_features = load_ckpt(ckpt_dir, "features_train.pkl") if args.resume else None
    if df_features is None:
        df_features = build_features_for_pairs(df_pairs, df_s1, df_pool, blocker=blocker_keep,
                                               use_cross_encoder=not args.no_cross_encoder)
        save_ckpt(ckpt_dir, "features_train.pkl", df_features)

    # =================== TRAIN XGBOOST ===================
    strata = df_features['source1_entity_id'].map(
        df_s1.set_index('entity_id')['country_norm']).fillna('').values
    models, thresholds, feature_cols, oof_proba = train_xgboost(
        df_features, df_features['is_true_match'], df_features['source1_entity_id'],
        strata, args.output_dir, gt, seeds, args.stratify, args.no_spw
    )

    # =================== LLM SNIPER ===================
    global_thresh = thresholds['default']
    if not args.skip_llm:
        print("\n========== LLM SNIPER (Borderline Arbitration) ==========")
        band_lo = args.band_lo if args.band_lo is not None else max(0.05, global_thresh - 0.15)
        band_hi = args.band_hi if args.band_hi is not None else min(0.97, global_thresh + 0.15)
        borderline_mask = (oof_proba >= band_lo) & (oof_proba <= band_hi)
        n_borderline = int(borderline_mask.sum())
        print(f"Borderline pairs ({band_lo:.2f}-{band_hi:.2f}): {n_borderline}")

        if 0 < n_borderline < 5000:
            try:
                df_border = df_features[borderline_mask].copy()
                df_border['xgb_prob'] = oof_proba[borderline_mask]
                df_border = df_border.merge(df_s1[['entity_id', 'clean_name', 'clean_address']],
                                            left_on='source1_entity_id', right_on='entity_id', how='left')
                df_border.rename(columns={'clean_name': 'name_s1', 'clean_address': 'addr_s1'}, inplace=True)
                df_border = df_border.merge(df_pool[['entity_id', 'clean_name', 'clean_address']],
                                            left_on='candidate_entity_id', right_on='entity_id', how='left')
                df_border.rename(columns={'clean_name': 'name_cand', 'clean_address': 'addr_cand'}, inplace=True)

                sniper = LLMSniper()
                llm_decisions = sniper.arbitrate(df_border)

                y_true_dict = build_true_dict(gt)
                s1_arr = df_features['source1_entity_id'].values
                cand_arr = df_features['candidate_entity_id'].values

                score_base = macro_score_from_proba(y_true_dict, s1_arr, cand_arr, oof_proba, global_thresh)
                llm_proba = oof_proba.copy()
                llm_proba[borderline_mask] = [0.99 if d == 1 else 0.01 for d in llm_decisions]
                score_llm = macro_score_from_proba(y_true_dict, s1_arr, cand_arr, llm_proba, global_thresh)

                print(f"LLM arbitration: base OOF F-0.5 {score_base:.4f} -> with LLM {score_llm:.4f} "
                      f"(delta {score_llm - score_base:+.4f})")
                if score_llm < score_base:
                    print(">>> LLM HURTS on validation - run predict.py with --skip_llm.")
            except Exception as e:
                print(f"LLM Sniper failed: {e}. Using XGBoost threshold only.")
        else:
            print(f"Skipping LLM (borderline pairs: {n_borderline})")

    print("\n========== TRAINING COMPLETE ==========")
    print("Next: python error_analysis.py --data_dir <train> --output_dir <out>  # bucket your errors")
    print("Then: python predict.py --test_dir <test> --model_path <out>/xgb_model.pkl --resume --skip_llm")

if __name__ == "__main__":
    main()
