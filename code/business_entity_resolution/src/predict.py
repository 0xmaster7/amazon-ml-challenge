"""
predict.py — Test-set inference pipeline.

Loads the trained XGBoost model, runs blocking + feature engineering on test data,
applies the learned threshold, runs LLM sniper on borderline pairs, and outputs
both matching_results.tsv and candidate_pairs.tsv in the exact competition format.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Path to dataset/test/")
    parser.add_argument("--model_path", type=str, required=True, help="Path to xgb_model.pkl")
    parser.add_argument("--output_dir", type=str, default="../../output")
    parser.add_argument("--skip_llm", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # =================== LOAD MODEL ===================
    print("========== LOADING MODEL ==========")
    with open(args.model_path, 'rb') as f:
        saved = pickle.load(f)
    models = saved['models']
    threshold = saved['threshold']
    feature_cols = saved['features']
    print(f"Loaded {len(models)} models. Threshold: {threshold}")
    print(f"Features: {feature_cols}")

    # =================== LOAD & CLEAN TEST DATA ===================
    print("\n========== LOADING TEST DATA ==========")
    df_s1 = process_dataframe(pd.read_csv(os.path.join(args.test_dir, "test_source1.tsv"), sep="\t"))
    df_s2 = process_dataframe(pd.read_csv(os.path.join(args.test_dir, "test_source2.tsv"), sep="\t"))
    df_s3 = process_dataframe(pd.read_csv(os.path.join(args.test_dir, "test_source3.tsv"), sep="\t"))
    df_pool = pd.concat([df_s2, df_s3]).reset_index(drop=True)
    
    all_s1_ids = set(df_s1['entity_id'].values)
    all_pool_ids = set(df_pool['entity_id'].values)
    print(f"Test S1 entities: {len(all_s1_ids)}")
    print(f"Test S2+S3 pool: {len(all_pool_ids)}")

    # =================== BLOCKING ===================
    print("\n========== STAGE 1: LAYERED BLOCKING ==========")
    blocker = LayeredBlocker()
    blocker.layer1_exact_key_blocking(df_s1, df_s2, df_s3)
    # blocker.layer2_minhash_lsh(df_s1, df_s2, df_s3)
    blocker.layer3_semantic_embeddings(df_s1, df_s2, df_s3)
    blocker.layer4_address_only(df_s1, df_s2, df_s3)

    df_pairs = blocker.export_candidate_pairs(os.path.join(args.output_dir, "candidate_pairs.tsv"))
    
    # Store candidate set for subset validation later
    candidate_set = set()
    for _, row in df_pairs.iterrows():
        candidate_set.add((row['source1_entity_id'], row['candidate_entity_id']))

    # =================== FEATURE ENGINEERING ===================
    print("\n========== STAGE 2: FEATURE ENGINEERING ==========")
    df_features = build_features_for_pairs(df_pairs, df_s1, df_pool, blocker=blocker)

    # =================== ENSEMBLE PREDICTION ===================
    print("\n========== PREDICTION ==========")
    X_test = df_features[feature_cols]
    
    # Average predictions across all fold models (ensemble)
    all_proba = np.zeros(len(X_test))
    for model in models:
        all_proba += model.predict_proba(X_test)[:, 1]
    all_proba /= len(models)
    
    df_features['xgb_prob'] = all_proba
    
    # =================== LLM SNIPER ===================
    df_features['final_pred'] = (all_proba > threshold).astype(int)
    
    if not args.skip_llm:
        print("\n========== LLM SNIPER (Borderline Arbitration) ==========")
        borderline_mask = (all_proba >= 0.35) & (all_proba <= 0.65)
        n_borderline = borderline_mask.sum()
        print(f"Borderline pairs (0.35-0.65): {n_borderline}")
        
        if n_borderline > 0 and n_borderline < 5000:
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

    # =================== BUILD matching_results.tsv ===================
    print("\n========== BUILDING matching_results.tsv ==========")
    
    # Get predicted matches
    matched_pairs = df_features[df_features['final_pred'] == 1][['source1_entity_id', 'candidate_entity_id']]
    
    # Group by S1 entity
    if len(matched_pairs) > 0:
        results = matched_pairs.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ','.join(sorted(set(x)))
        ).reset_index()
        results.rename(columns={'candidate_entity_id': 'matched_entity_ids'}, inplace=True)
    else:
        results = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])
    
    # CRITICAL: Every S1 entity must appear, even singletons (with empty matched_entity_ids)
    all_s1 = pd.DataFrame({'source1_entity_id': sorted(all_s1_ids)})
    results = all_s1.merge(results, on='source1_entity_id', how='left')
    results['matched_entity_ids'] = results['matched_entity_ids'].fillna('')
    
    # Save
    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    results.to_csv(matching_path, sep='\t', index=False)
    print(f"Saved {matching_path}")
    print(f"  Total S1 entities: {len(results)}")
    print(f"  Entities with matches: {(results['matched_entity_ids'] != '').sum()}")
    print(f"  Singletons (empty): {(results['matched_entity_ids'] == '').sum()}")

    # =================== REBUILD candidate_pairs.tsv (Competition Format) ===================
    # The competition wants: source1_entity_id \t candidate_entity_ids
    # One row per S1 entity, even if empty
    print("\n========== REBUILDING candidate_pairs.tsv (Competition Format) ==========")
    if len(df_pairs) > 0:
        cand_grouped = df_pairs.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ','.join(sorted(set(x)))
        ).reset_index()
        cand_grouped.rename(columns={'candidate_entity_id': 'candidate_entity_ids'}, inplace=True)
    else:
        cand_grouped = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_ids'])
    
    cand_grouped = all_s1.merge(cand_grouped, on='source1_entity_id', how='left')
    cand_grouped['candidate_entity_ids'] = cand_grouped['candidate_entity_ids'].fillna('')
    
    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    cand_grouped.to_csv(cand_path, sep='\t', index=False)
    print(f"Saved {cand_path}")

    # =================== VALIDATION ===================
    print("\n========== SELF-VALIDATION ==========")
    errors = validate_outputs(results, cand_grouped, all_s1_ids, all_pool_ids)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print("✅ VALIDATION PASSED — safe to submit!")

    print("\n========== PREDICTION PIPELINE COMPLETE ==========")


def validate_outputs(matching_df, candidate_df, all_s1_ids, all_pool_ids):
    """
    Replicates the checks from utils/validate_submission.py:
    1. Every S1 entity has exactly one row in matching_results.tsv
    2. No duplicate entity IDs within a single ID list
    3. IDs only reference S2/S3 entities that exist in the test set
    4. matching_results is a subset of candidate_pairs
    """
    errors = []
    
    # Check 1: Every S1 entity present
    matching_s1 = set(matching_df['source1_entity_id'].values)
    missing = all_s1_ids - matching_s1
    if missing:
        errors.append(f"Missing {len(missing)} S1 entities from matching_results.tsv")
    
    extra = matching_s1 - all_s1_ids
    if extra:
        errors.append(f"Found {len(extra)} extra S1 entities not in test set")
    
    # Check for duplicate S1 rows
    if matching_df['source1_entity_id'].duplicated().any():
        errors.append("Duplicate source1_entity_id rows in matching_results.tsv")
    
    # Check 2: No duplicate IDs within a list, and all IDs exist in pool
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
                break  # Don't spam
    
    # Check 3: matching_results is a SUBSET of candidate_pairs
    # Build candidate lookup
    cand_lookup = {}
    for _, row in candidate_df.iterrows():
        s1 = row['source1_entity_id']
        cands = row['candidate_entity_ids']
        if cands == '':
            cand_lookup[s1] = set()
        else:
            cand_lookup[s1] = set(cands.split(','))
    
    for _, row in matching_df.iterrows():
        s1 = row['source1_entity_id']
        ids_str = row['matched_entity_ids']
        if ids_str == '':
            continue
        matched_ids = set(ids_str.split(','))
        candidates = cand_lookup.get(s1, set())
        not_in_candidates = matched_ids - candidates
        if not_in_candidates:
            errors.append(f"matched IDs {not_in_candidates} for {s1} not in candidate_pairs (SUBSET VIOLATION)")
    
    return errors


if __name__ == "__main__":
    main()
