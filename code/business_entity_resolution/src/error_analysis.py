"""
error_analysis.py — Comprehensive Post-Training Error Analysis

Performs deep diagnostics on the training/validation data:
1. Blocking Recall Ceiling (hard cap on potential performance)
2. False Merges (False Positives) — franchise traps, PIN mismatches, top 20 cases
3. False Negatives (Missed True Matches) — mojibake, missing addresses, top 20 cases
4. Probability Score Distributions (TP, FP, TN, FN) to guide LLM borderline tuning
5. Country-Stratified F-0.5 macro score evaluation (India vs US)
"""
import argparse
import os
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_cleaner import process_dataframe
from blocker import LayeredBlocker
from feature_engineering import build_features_for_pairs
from train import f_05_per_entity


def main():
    parser = argparse.ArgumentParser(description="Perform error analysis on trained model and candidate pairs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--model_path", type=str, default="../../output/xgb_model.pkl", help="Path to trained model pickle")
    parser.add_argument("--output_dir", type=str, default="../../output", help="Directory with candidate_pairs.tsv if already built")
    parser.add_argument("--test_mode", action="store_true", help="Run on a small subset for quick debugging")
    args = parser.parse_args()

    print("=" * 70)
    print("        AMAZON ML CHALLENGE: COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 70)

    # 1. LOAD MODEL
    print("\n[1/6] Loading model from:", args.model_path)
    with open(args.model_path, 'rb') as f:
        saved = pickle.load(f)
    models = saved['models']
    threshold = saved.get('threshold', 0.5)
    country_thresholds = saved.get('country_thresholds', {})
    feature_cols = saved['features']
    print(f"Loaded {len(models)} models. Global Threshold: {threshold}")
    if country_thresholds:
        print(f"Country Stratified Thresholds: {country_thresholds}")

    # 2. LOAD DATA
    print("\n[2/6] Loading and cleaning datasets from:", args.data_dir)
    nrows = 2000 if args.test_mode else None
    df_s1 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source1.tsv"), sep='\t', nrows=nrows))
    df_s2 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source2.tsv"), sep='\t', nrows=nrows))
    df_s3 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source3.tsv"), sep='\t', nrows=nrows))
    df_pool = pd.concat([df_s2, df_s3]).reset_index(drop=True)
    gt = pd.read_csv(os.path.join(args.data_dir, "train_ground_truth.tsv"), sep='\t')

    # Prepare ground truth exploded pairs
    singleton_mask = gt['matched_entity_ids'].isna() | (gt['matched_entity_ids'] == "")
    gt_non_empty = gt[~singleton_mask].copy()
    gt_exploded = gt_non_empty.assign(
        matched_entity_ids=gt_non_empty['matched_entity_ids'].str.split(',')
    ).explode('matched_entity_ids')
    gt_pairs_set = set(zip(gt_exploded['source1_entity_id'], gt_exploded['matched_entity_ids']))
    total_true_matches = len(gt_pairs_set)

    # 3. BLOCKING CANDIDATE GENERATION
    print("\n[3/6] Running Stage 1 Blocker (reusing precomputed embeddings if available)...")
    blocker = LayeredBlocker()
    blocker.layer1_exact_key_blocking(df_s1, df_s2, df_s3)
    blocker.layer3_semantic_embeddings(df_s1, df_s2, df_s3)
    blocker.layer4_address_only(df_s1, df_s2, df_s3)

    os.makedirs(args.output_dir, exist_ok=True)
    df_pairs = blocker.export_candidate_pairs(os.path.join(args.output_dir, "error_analysis_candidates.tsv"))
    cand_pairs_set = set(zip(df_pairs['source1_entity_id'], df_pairs['candidate_entity_id']))

    # -------------------------------------------------------------
    # DIAGNOSTIC 1: BLOCKING RECALL CEILING
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. BLOCKING RECALL CEILING ANALYSIS")
    print("=" * 70)
    found_pairs = gt_pairs_set.intersection(cand_pairs_set)
    missed_pairs = gt_pairs_set - cand_pairs_set
    recall_ceiling = (len(found_pairs) / total_true_matches * 100) if total_true_matches > 0 else 0

    print(f"Total True Match Pairs in Ground Truth:   {total_true_matches}")
    print(f"Matches Successfully Found by Blocker:    {len(found_pairs)} ({recall_ceiling:.2f}%)")
    print(f"Matches MISSED ENTIRELY by Blocker:       {len(missed_pairs)} ({100 - recall_ceiling:.2f}%)")
    print(f"Total Generated Candidates:               {len(df_pairs)}")
    print(f"Avg Candidates per S1 Entity:             {len(df_pairs) / max(len(df_s1), 1):.2f}")
    print("\n>>> NOTE: Missed pairs form the HARD CEILING. No classifier can ever recover them.")

    if missed_pairs and len(df_s1) > 0:
        print("\nSample of Missed True Matches (first 5):")
        s1_lookup = df_s1.set_index('entity_id')
        pool_lookup = df_pool.set_index('entity_id')
        for i, (s1_id, cand_id) in enumerate(list(missed_pairs)[:5]):
            s1_row = s1_lookup.loc[s1_id] if s1_id in s1_lookup.index else None
            cand_row = pool_lookup.loc[cand_id] if cand_id in pool_lookup.index else None
            print(f"  [{i+1}] S1 ({s1_id}):   {s1_row['clean_name'] if s1_row is not None else 'N/A'} | {s1_row['clean_address'] if s1_row is not None else 'N/A'}")
            print(f"      Cand ({cand_id}): {cand_row['clean_name'] if cand_row is not None else 'N/A'} | {cand_row['clean_address'] if cand_row is not None else 'N/A'}")

    # 4. FEATURE ENGINEERING & PREDICTION
    print("\n[4/6] Building features for candidate pairs...")
    # Label pairs with ground truth
    df_pairs['is_true_match'] = df_pairs.apply(
        lambda r: 1 if (r['source1_entity_id'], r['candidate_entity_id']) in gt_pairs_set else 0, axis=1
    )

    df_features = build_features_for_pairs(df_pairs, df_s1, df_pool, blocker=blocker)

    print("\n[5/6] Generating ensemble predictions...")
    X = df_features[feature_cols]
    all_proba = np.zeros(len(X))
    for m in models:
        all_proba += m.predict_proba(X)[:, 1]
    all_proba /= len(models)
    df_features['xgb_prob'] = all_proba

    # Apply stratified thresholds if available
    final_preds = np.zeros(len(df_features), dtype=int)
    if 'country_s1' in df_features.columns and country_thresholds:
        for country, c_thresh in country_thresholds.items():
            mask = df_features['country_s1'].str.lower() == country
            final_preds[mask] = (all_proba[mask] > c_thresh).astype(int)
        other_mask = ~df_features['country_s1'].str.lower().isin(country_thresholds.keys())
        final_preds[other_mask] = (all_proba[other_mask] > threshold).astype(int)
    else:
        final_preds = (all_proba > threshold).astype(int)
    df_features['pred'] = final_preds

    # -------------------------------------------------------------
    # DIAGNOSTIC 2: OVERALL ACCURACY & F-0.5 SCORE
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. OVERALL VALIDATION F-0.5 MACRO SCORE")
    print("=" * 70)
    # Build per-entity evaluation dicts
    all_s1_ids = df_s1['entity_id'].unique()
    y_true_dict = {}
    y_pred_dict = {}
    for s1_id in all_s1_ids:
        gt_row = gt[gt['source1_entity_id'] == s1_id]
        if len(gt_row) > 0 and pd.notna(gt_row.iloc[0]['matched_entity_ids']) and gt_row.iloc[0]['matched_entity_ids'] != '':
            y_true_dict[s1_id] = set(str(gt_row.iloc[0]['matched_entity_ids']).split(','))
        else:
            y_true_dict[s1_id] = set()
        y_pred_dict[s1_id] = set()

    for _, row in df_features[df_features['pred'] == 1].iterrows():
        s1 = row['source1_entity_id']
        cand = row['candidate_entity_id']
        if s1 in y_pred_dict:
            y_pred_dict[s1].add(cand)

    macro_f05 = f_05_per_entity(y_true_dict, y_pred_dict)
    print(f">>> Full Training Set Macro F-0.5 Score: {macro_f05:.4f}")

    # -------------------------------------------------------------
    # DIAGNOSTIC 3: CLASSIFIER CONFUSION MATRIX
    # -------------------------------------------------------------
    tp = ((df_features['pred'] == 1) & (df_features['is_true_match'] == 1)).sum()
    fp = ((df_features['pred'] == 1) & (df_features['is_true_match'] == 0)).sum()
    fn = ((df_features['pred'] == 0) & (df_features['is_true_match'] == 1)).sum()
    tn = ((df_features['pred'] == 0) & (df_features['is_true_match'] == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\nPair-Level Confusion Matrix:")
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}  <-- Penalized 2x heavily by F-0.5!")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  Pair Precision:       {precision:.4f}")
    print(f"  Pair Recall:          {recall:.4f}")

    # Attach text columns for qualitative error review
    s1_sub = df_s1[['entity_id', 'business_name', 'business_address', 'clean_name', 'clean_address', 'extracted_pin', 'country']].copy()
    pool_sub = df_pool[['entity_id', 'business_name', 'business_address', 'clean_name', 'clean_address', 'extracted_pin', 'country']].copy()

    df_full = df_features.merge(s1_sub, left_on='source1_entity_id', right_on='entity_id', how='left', suffixes=('', '_s1_meta'))
    df_full = df_full.merge(pool_sub, left_on='candidate_entity_id', right_on='entity_id', how='left', suffixes=('_s1', '_cand'))

    # -------------------------------------------------------------
    # DIAGNOSTIC 4: FALSE POSITIVES (FALSE MERGES)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. FALSE POSITIVES (FALSE MERGES) ROOT CAUSE ANALYSIS")
    print("=" * 70)
    fp_df = df_full[(df_full['pred'] == 1) & (df_full['is_true_match'] == 0)].copy()

    if len(fp_df) > 0:
        # Franchise/Chain trap: Same clean name, different address
        chain_trap = fp_df[(fp_df['clean_name_s1'] == fp_df['clean_name_cand']) & (fp_df['clean_address_s1'] != fp_df['clean_address_cand'])]
        # PIN mismatch
        pin_mismatch = fp_df[(fp_df['extracted_pin_s1'] != "") & (fp_df['extracted_pin_cand'] != "") & (fp_df['extracted_pin_s1'] != fp_df['extracted_pin_cand'])]
        # Country mismatch
        country_mismatch = fp_df[fp_df['country_s1'] != fp_df['country_cand']]

        print(f"Total False Positives:                    {len(fp_df)}")
        print(f"  - Chain/Franchise Trap (same name, diff addr): {len(chain_trap)} ({len(chain_trap)/len(fp_df)*100:.1f}%)")
        print(f"  - Different Pincode/ZIP:                       {len(pin_mismatch)} ({len(pin_mismatch)/len(fp_df)*100:.1f}%)")
        print(f"  - Country Mismatch:                            {len(country_mismatch)} ({len(country_mismatch)/len(fp_df)*100:.1f}%)")

        print("\nTop 10 Worst False Positives (Highest Model Confidence):")
        worst_fps = fp_df.sort_values(by='xgb_prob', ascending=False).head(10)
        for i, (_, row) in enumerate(worst_fps.iterrows()):
            print(f"  [{i+1}] Prob: {row['xgb_prob']:.4f}")
            print(f"      S1:   {row['business_name_s1']} | {row['business_address_s1']}")
            print(f"      Cand: {row['business_name_cand']} | {row['business_address_cand']}")
    else:
        print("Zero False Positives found! (Perfect precision)")

    # -------------------------------------------------------------
    # DIAGNOSTIC 5: FALSE NEGATIVES (MISSED MATCHES)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. FALSE NEGATIVES (MISSED RETRIEVED MATCHES) ROOT CAUSE ANALYSIS")
    print("=" * 70)
    fn_df = df_full[(df_full['pred'] == 0) & (df_full['is_true_match'] == 1)].copy()

    if len(fn_df) > 0:
        missing_addr = fn_df[(fn_df['clean_address_s1'] == "") | (fn_df['clean_address_cand'] == "")]
        low_name_sim = fn_df[fn_df.get('name_jaro_winkler', pd.Series(1, index=fn_df.index)) < 0.6]

        print(f"Total False Negatives (among candidates):  {len(fn_df)}")
        print(f"  - Missing Address in S1 or Cand:         {len(missing_addr)} ({len(missing_addr)/len(fn_df)*100:.1f}%)")
        print(f"  - Low Name Similarity (DBA/alias mismatch): {len(low_name_sim)} ({len(low_name_sim)/len(fn_df)*100:.1f}%)")

        print("\nTop 10 Worst False Negatives (Lowest Model Confidence on True Matches):")
        worst_fns = fn_df.sort_values(by='xgb_prob', ascending=True).head(10)
        for i, (_, row) in enumerate(worst_fns.iterrows()):
            print(f"  [{i+1}] Prob: {row['xgb_prob']:.4f}")
            print(f"      S1:   {row['business_name_s1']} | {row['business_address_s1']}")
            print(f"      Cand: {row['business_name_cand']} | {row['business_address_cand']}")
    else:
        print("Zero False Negatives among retrieved candidates!")

    # -------------------------------------------------------------
    # DIAGNOSTIC 6: PROBABILITY DISTRIBUTION (Tuning LLM Band)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. PROBABILITY SCORE DISTRIBUTION (Guides LLM Arbitration Band)")
    print("=" * 70)
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    df_features['prob_bucket'] = pd.cut(df_features['xgb_prob'], bins=bins, include_lowest=True)

    print(f"{'Prob Range':<15} | {'True Matches':<15} | {'Non-Matches':<15} | {'Match %':<10}")
    print("-" * 62)
    for b in pd.Series(df_features['prob_bucket'].unique()).dropna().sort_values():
        subset = df_features[df_features['prob_bucket'] == b]
        m_count = (subset['is_true_match'] == 1).sum()
        nm_count = (subset['is_true_match'] == 0).sum()
        pct = (m_count / len(subset) * 100) if len(subset) > 0 else 0
        print(f"{str(b):<15} | {m_count:<15} | {nm_count:<15} | {pct:<10.1f}%")

    # -------------------------------------------------------------
    # DIAGNOSTIC 7: PER-COUNTRY BREAKDOWN (India vs US)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. PER-COUNTRY F-0.5 BREAKDOWN (India vs US)")
    print("=" * 70)
    if 'country_s1' in df_features.columns:
        for c in ['india', 'us']:
            c_s1_ids = df_s1[df_s1['country'].str.lower() == c]['entity_id'].unique()
            c_true = {k: y_true_dict[k] for k in c_s1_ids if k in y_true_dict}
            c_pred = {k: y_pred_dict.get(k, set()) for k in c_s1_ids}
            c_score = f_05_per_entity(c_true, c_pred)
            print(f"  {c.upper()} Entities: {len(c_s1_ids)} | Macro F-0.5: {c_score:.4f}")

    print("\n" + "=" * 70)
    print("        ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
