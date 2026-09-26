"""
predict.py — Test-set inference pipeline.

Blocking (6 layers, unioned) -> feature engineering -> seed/fold-ensembled
XGBoost probabilities -> per-stratum OOF-calibrated thresholds -> LLM
arbitration on the borderline band -> post-processing conflict resolution
-> matching_results.tsv + candidate_pairs.tsv with self-validation.
"""
import pandas as pd
import numpy as np
import os
import argparse
import pickle
from data_cleaner import process_dataframe
from blocker import LayeredBlocker
from feature_engineering import build_features_for_pairs
from llm_sniper import LLMSniper


# ============================================================
# CHECKPOINTING
# ============================================================
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


def row_countries(df_features, df_s1):
    m = df_s1.set_index('entity_id')['country_norm']
    return df_features['source1_entity_id'].map(m).fillna('').values


def apply_thresholds(row_country, proba, thresholds):
    if len(thresholds) == 1:
        return (proba > thresholds['default']).astype(int)
    t = np.array([thresholds.get(c, thresholds['default']) for c in row_country])
    return (proba > t).astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../../output")
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--no_cross_encoder", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--embedder2", type=str, default=None,
                        help="Must match train.py if it was used there.")
    parser.add_argument("--band_lo", type=float, default=None)
    parser.add_argument("--band_hi", type=float, default=None)
    parser.add_argument("--no_conflict_resolution", action="store_true",
                        help="Disable the one-S1-per-pool-ID post-processing pass. "
                             "Use this if train.py's GT conflict check warned.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # =================== LOAD MODEL ===================
    print("========== LOADING MODEL ==========")
    with open(args.model_path, 'rb') as f:
        saved = pickle.load(f)
    models = saved['models']
    thresholds = saved.get('thresholds', {'default': saved.get('threshold', 0.5)})
    feature_cols = saved['features']
    print(f"Loaded {len(models)} models. Thresholds: {thresholds}")

    # =================== LOAD & CLEAN TEST DATA ===================
    print("\n========== LOADING TEST DATA ==========")
    cleaned = load_ckpt(ckpt_dir, "cleaned_test.pkl") if args.resume else None
    if cleaned is not None:
        df_s1, df_s2, df_s3 = cleaned
    else:
        df_s1 = process_dataframe(pd.read_csv(os.path.join(args.test_dir, "test_source1.tsv"), sep="\t"))
        df_s2 = process_dataframe(pd.read_csv(os.path.join(args.test_dir, "test_source2.tsv"), sep="\t"))
        df_s3 = process_dataframe(pd.read_csv(os.path.join(args.test_dir, "test_source3.tsv"), sep="\t"))
        save_ckpt(ckpt_dir, "cleaned_test.pkl", (df_s1, df_s2, df_s3))
    df_pool = pd.concat([df_s2, df_s3]).reset_index(drop=True)

    all_s1_ids = set(df_s1['entity_id'].values)
    all_pool_ids = set(df_pool['entity_id'].values)
    print(f"Test S1 entities: {len(all_s1_ids)} | Test S2+S3 pool: {len(all_pool_ids)}")
    print(f"Countries in test: {sorted(df_s1['country'].dropna().unique())}")

    # =================== BLOCKING ===================
    print("\n========== STAGE 1: LAYERED BLOCKING ==========")
    blocker_keep = None
    df_pairs = load_ckpt(ckpt_dir, "pairs_test.pkl") if args.resume else None
    if df_pairs is None:
        blocker_keep = LayeredBlocker()
        blocker_keep.layer1_exact_key_blocking(df_s1, df_s2, df_s3)
        blocker_keep.layer2_tfidf_blocking(df_s1, df_s2, df_s3)
        blocker_keep.layer3_semantic_embeddings(df_s1, df_s2, df_s3)
        if args.embedder2:
            blocker_keep.layer3_semantic_embeddings(df_s1, df_s2, df_s3, model_name=args.embedder2)
        blocker_keep.layer4_address_only(df_s1, df_s2, df_s3)
        blocker_keep.layer4b_near_exact_address(df_s1, df_s2, df_s3)
        blocker_keep.layer5_phone_key_blocking(df_s1, df_s2, df_s3)
        df_pairs = blocker_keep.export_candidate_pairs(os.path.join(args.output_dir, "candidate_pairs.tsv"))
        save_ckpt(ckpt_dir, "pairs_test.pkl", df_pairs)

    # =================== FEATURE ENGINEERING ===================
    print("\n========== STAGE 2: FEATURE ENGINEERING ==========")
    df_features = load_ckpt(ckpt_dir, "features_test.pkl") if args.resume else None
    if df_features is None:
        df_features = build_features_for_pairs(df_pairs, df_s1, df_pool, blocker=blocker_keep,
                                               use_cross_encoder=not args.no_cross_encoder)
        save_ckpt(ckpt_dir, "features_test.pkl", df_features)

    # =================== ENSEMBLE PREDICTION ===================
    print("\n========== PREDICTION ==========")
    X_test = df_features[feature_cols]

    all_proba = np.zeros(len(X_test))
    for model in models:
        all_proba += model.predict_proba(X_test)[:, 1]
    all_proba /= len(models)
    df_features['xgb_prob'] = all_proba

    rc = row_countries(df_features, df_s1)
    df_features['final_pred'] = apply_thresholds(rc, all_proba, thresholds)

    # =================== LLM SNIPER ===================
    global_thresh = thresholds['default']
    if not args.skip_llm:
        print("\n========== LLM SNIPER (Borderline Arbitration) ==========")
        band_lo = args.band_lo if args.band_lo is not None else max(0.05, global_thresh - 0.15)
        band_hi = args.band_hi if args.band_hi is not None else min(0.97, global_thresh + 0.15)
        borderline_mask = (all_proba >= band_lo) & (all_proba <= band_hi)
        n_borderline = int(borderline_mask.sum())
        print(f"Borderline pairs ({band_lo:.2f}-{band_hi:.2f}): {n_borderline}")

        if 0 < n_borderline < 5000:
            try:
                df_border = df_features[borderline_mask].copy()
                df_border = df_border.merge(df_s1[['entity_id', 'clean_name', 'clean_address']],
                                            left_on='source1_entity_id', right_on='entity_id', how='left')
                df_border.rename(columns={'clean_name': 'name_s1', 'clean_address': 'addr_s1'}, inplace=True)
                df_border = df_border.merge(df_pool[['entity_id', 'clean_name', 'clean_address']],
                                            left_on='candidate_entity_id', right_on='entity_id', how='left')
                df_border.rename(columns={'clean_name': 'name_cand', 'clean_address': 'addr_cand'}, inplace=True)
                sniper = LLMSniper()
                llm_decisions = sniper.arbitrate(df_border)
                df_features.loc[borderline_mask, 'final_pred'] = llm_decisions
                print(f"LLM decided {sum(1 for d in llm_decisions if d == 1)} borderline pairs are MATCH.")
            except Exception as e:
                print(f"LLM Sniper failed: {e}. Using XGBoost threshold only.")

    # =================== POST-PROCESSING: CONFLICT RESOLUTION ============
    # A pool record should not be claimed by two different S1 entities
    # (train.py verifies this assumption against ground truth). Keep only
    # the highest-confidence claim. Cheap precision boost a per-pair
    # classifier cannot do on its own.
    if not args.no_conflict_resolution:
        claimed = df_features[df_features['final_pred'] == 1]
        conflicted = claimed['candidate_entity_id'].value_counts()
        conflicted = conflicted[conflicted > 1]
        n_dropped = 0
        for cid in conflicted.index:
            rows = claimed[claimed['candidate_entity_id'] == cid]
            keep_idx = rows['xgb_prob'].idxmax()
            drop_idx = rows.index.difference([keep_idx])
            df_features.loc[drop_idx, 'final_pred'] = 0
            n_dropped += len(drop_idx)
        print(f"\nConflict resolution: {len(conflicted)} pool IDs were double-claimed; "
              f"dropped {n_dropped} lower-confidence claims.")

    # =================== BUILD OUTPUTS ===================
    print("\n========== BUILDING matching_results.tsv ==========")
    matched_pairs = df_features[df_features['final_pred'] == 1][['source1_entity_id', 'candidate_entity_id']]

    if len(matched_pairs) > 0:
        results = matched_pairs.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ','.join(sorted(set(x)))).reset_index()
        results.rename(columns={'candidate_entity_id': 'matched_entity_ids'}, inplace=True)
    else:
        results = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])

    all_s1 = pd.DataFrame({'source1_entity_id': sorted(all_s1_ids)})
    results = all_s1.merge(results, on='source1_entity_id', how='left')
    results['matched_entity_ids'] = results['matched_entity_ids'].fillna('')

    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    results.to_csv(matching_path, sep='\t', index=False)
    print(f"Saved {matching_path}: {(results['matched_entity_ids'] != '').sum()} with matches, "
          f"{(results['matched_entity_ids'] == '').sum()} singletons")

    print("\n========== REBUILDING candidate_pairs.tsv ==========")
    if len(df_pairs) > 0:
        cand_grouped = df_pairs.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ','.join(sorted(set(x)))).reset_index()
        cand_grouped.rename(columns={'candidate_entity_id': 'candidate_entity_ids'}, inplace=True)
    else:
        cand_grouped = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_ids'])
    cand_grouped = all_s1.merge(cand_grouped, on='source1_entity_id', how='left')
    cand_grouped['candidate_entity_ids'] = cand_grouped['candidate_entity_ids'].fillna('')
    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    cand_grouped.to_csv(cand_path, sep='\t', index=False)
    print(f"Saved {cand_path}")

    print("\n========== SELF-VALIDATION ==========")
    errors = validate_outputs(results, cand_grouped, all_s1_ids, all_pool_ids)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print("✅ VALIDATION PASSED — now run the official utils/validate_submission.py too.")

    print("\n========== PREDICTION COMPLETE ==========")


def validate_outputs(matching_df, candidate_df, all_s1_ids, all_pool_ids):
    errors = []
    matching_s1 = set(matching_df['source1_entity_id'].values)
    missing = all_s1_ids - matching_s1
    if missing:
        errors.append(f"Missing {len(missing)} S1 entities from matching_results.tsv")
    extra = matching_s1 - all_s1_ids
    if extra:
        errors.append(f"Found {len(extra)} extra S1 entities not in test set")
    if matching_df['source1_entity_id'].duplicated().any():
        errors.append("Duplicate source1_entity_id rows in matching_results.tsv")
    for _, row in matching_df.iterrows():
        ids_str = row['matched_entity_ids']
        if ids_str == '':
            continue
        ids = ids_str.split(',')
        if len(ids) != len(set(ids)):
            errors.append(f"Duplicate IDs in matched_entity_ids for {row['source1_entity_id']}")
        for eid in ids:
            if eid not in all_pool_ids:
                errors.append(f"ID {eid} in matching_results not found in test S2/S3 pool")
                break
    cand_lookup = {}
    for _, row in candidate_df.iterrows():
        cands = row['candidate_entity_ids']
        cand_lookup[row['source1_entity_id']] = set(cands.split(',')) if cands else set()
    for _, row in matching_df.iterrows():
        ids_str = row['matched_entity_ids']
        if ids_str == '':
            continue
        not_in = set(ids_str.split(',')) - cand_lookup.get(row['source1_entity_id'], set())
        if not_in:
            errors.append(f"matched IDs {not_in} for {row['source1_entity_id']} not in candidate_pairs (SUBSET VIOLATION)")
    return errors


if __name__ == "__main__":
    main()
