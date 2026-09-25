import pandas as pd
import numpy as np
import os
import argparse
import pickle
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from data_cleaner import process_dataframe
from blocker import LayeredBlocker
from feature_engineering import build_features_for_pairs
from llm_sniper import LLMSniper

# ============================================================
# F-0.5 SCORER — PER-ENTITY MACRO AVERAGE (Competition Formula)
# ============================================================
def f_05_per_entity(y_true_dict, y_pred_dict):
    """
    Computes the competition's exact F-0.5 macro-average.
    
    Args:
        y_true_dict: dict mapping source1_entity_id -> set of true matched_entity_ids (empty set for singletons)
        y_pred_dict: dict mapping source1_entity_id -> set of predicted matched_entity_ids (empty set for singletons)
    
    Returns:
        Macro-averaged F-0.5 score across all S1 entities.
    """
    scores = []
    for s1_id in y_true_dict:
        true_set = y_true_dict[s1_id]
        pred_set = y_pred_dict.get(s1_id, set())
        
        if len(true_set) == 0 and len(pred_set) == 0:
            # Singleton correctly predicted as empty -> 1.0
            scores.append(1.0)
        elif len(true_set) == 0 and len(pred_set) > 0:
            # Singleton with false predictions -> 0.0
            scores.append(0.0)
        elif len(true_set) > 0 and len(pred_set) == 0:
            # Has matches but predicted empty -> precision is undefined (0/0)
            # Recall = 0, so F-0.5 = 0.0
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


def build_entity_dicts(df_features, preds, s1_ids_all, gt):
    """
    Converts row-level predictions into per-entity sets for the macro scorer.
    """
    # Build true dict from ground truth
    y_true_dict = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or matched == '':
            y_true_dict[s1_id] = set()
        else:
            y_true_dict[s1_id] = set(str(matched).split(','))
    
    # Build pred dict from predictions
    y_pred_dict = {s1_id: set() for s1_id in y_true_dict}
    for idx, pred in enumerate(preds):
        if pred == 1:
            s1_id = df_features.iloc[idx]['source1_entity_id']
            cand_id = df_features.iloc[idx]['candidate_entity_id']
            if s1_id in y_pred_dict:
                y_pred_dict[s1_id].add(cand_id)
    
    return y_true_dict, y_pred_dict


# ============================================================
# XGBOOST TRAINING WITH CORRECT MACRO F-0.5 VALIDATION
# ============================================================
def train_xgboost(df_features, labels, groups, output_dir, gt):
    print("\n========== TRAINING XGBOOST ==========")
    exclude = ['source1_entity_id', 'candidate_entity_id', 'is_true_match', 'match_label']
    feature_cols = [c for c in df_features.columns if c not in exclude 
                    and df_features[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    X = df_features[feature_cols]
    y = labels
    
    gkf = GroupKFold(n_splits=5)
    models = []
    best_global_thresh = 0.5
    best_global_score = 0
    all_fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=50, random_state=42,
            tree_method="hist", device="cuda",
            scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        models.append(clf)
        
        preds_proba = clf.predict_proba(X_val)[:, 1]
        
        # Get the val subset of df_features for per-entity scoring
        df_val = df_features.iloc[val_idx].copy()
        
        # Fine-grained threshold sweep: 0.30 to 0.90 in steps of 0.02
        # Now using the CORRECT per-entity macro F-0.5
        best_thresh, best_score = 0.5, 0
        for thresh_int in range(30, 91, 2):
            thresh = thresh_int / 100.0
            preds = (preds_proba > thresh).astype(int)
            
            # Build per-entity dicts for this fold's validation set
            val_s1_ids = df_val['source1_entity_id'].unique()
            y_true_dict = {}
            y_pred_dict = {}
            for s1_id in val_s1_ids:
                # Get true matches from ground truth
                gt_row = gt[gt['source1_entity_id'] == s1_id]
                if len(gt_row) > 0:
                    matched = gt_row.iloc[0].get('matched_entity_ids', '')
                    if pd.isna(matched) or matched == '':
                        y_true_dict[s1_id] = set()
                    else:
                        y_true_dict[s1_id] = set(str(matched).split(','))
                else:
                    y_true_dict[s1_id] = set()
                y_pred_dict[s1_id] = set()
            
            # Fill predictions
            for i, (idx, row) in enumerate(df_val.iterrows()):
                if preds[i] == 1:
                    s1_id = row['source1_entity_id']
                    cand_id = row['candidate_entity_id']
                    if s1_id in y_pred_dict:
                        y_pred_dict[s1_id].add(cand_id)
            
            score = f_05_per_entity(y_true_dict, y_pred_dict)
            if score > best_score:
                best_score = score
                best_thresh = thresh
        
        print(f"Fold {fold+1} Best MACRO F-0.5: {best_score:.4f} at threshold: {best_thresh}")
        all_fold_scores.append(best_score)
        
        if best_score > best_global_score:
            best_global_score = best_score
            best_global_thresh = best_thresh
    
    avg_score = np.mean(all_fold_scores)
    print(f"\n>>> Average MACRO F-0.5 across folds: {avg_score:.4f}")
    print(f">>> Best single-fold MACRO F-0.5: {best_global_score:.4f} at threshold: {best_global_thresh}")
    
    # Save models, threshold, and feature list
    model_path = os.path.join(output_dir, "xgb_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'models': models, 'threshold': best_global_thresh, 'features': feature_cols}, f)
    print(f"Models saved to {model_path}")
    
    return models, best_global_thresh, feature_cols


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../../output")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM sniper (faster for debugging)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    nrows = 1000 if args.test_mode else None

    # =================== LOAD & CLEAN ===================
    print("========== LOADING DATA ==========")
    df_s1 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source1.tsv"), sep="\t", nrows=nrows))
    df_s2 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source2.tsv"), sep="\t", nrows=nrows))
    df_s3 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source3.tsv"), sep="\t", nrows=nrows))
    df_pool = pd.concat([df_s2, df_s3]).reset_index(drop=True)

    gt = pd.read_csv(os.path.join(args.data_dir, "train_ground_truth.tsv"), sep="\t")

    # =================== SINGLETON ANALYSIS ===================
    print("\n========== SINGLETON ANALYSIS ==========")
    singleton_mask = gt['matched_entity_ids'].isna() | (gt['matched_entity_ids'] == "")
    n_singletons = singleton_mask.sum()
    n_total = len(gt)
    print(f"Singletons (no true match): {n_singletons}/{n_total} ({100*n_singletons/n_total:.1f}%)")
    print(f"Multi-match entities: {n_total - n_singletons}")
    
    # Check max fan-out
    gt_non_empty = gt[~singleton_mask].copy()
    gt_non_empty['match_count'] = gt_non_empty['matched_entity_ids'].str.split(',').apply(len)
    max_fanout = gt_non_empty['match_count'].max()
    print(f"Max fan-out (most matches for one entity): {max_fanout}")
    print(f"FAISS k=50 >> {max_fanout}, so we won't truncate any true matches.")

    # =================== BLOCKING ===================
    print("\n========== STAGE 1: LAYERED BLOCKING ==========")
    blocker = LayeredBlocker()
    blocker.layer1_exact_key_blocking(df_s1, df_s2, df_s3)
    # blocker.layer2_minhash_lsh(df_s1, df_s2, df_s3)
    blocker.layer3_semantic_embeddings(df_s1, df_s2, df_s3)
    blocker.layer4_address_only(df_s1, df_s2, df_s3)

    df_pairs = blocker.export_candidate_pairs(os.path.join(args.output_dir, "candidate_pairs.tsv"))

    # =================== GROUND TRUTH LABELING ===================
    print("\n========== LABELING CANDIDATES ==========")
    gt_exploded = gt_non_empty.assign(
        matched_entity_ids=gt_non_empty['matched_entity_ids'].str.split(',')
    ).explode('matched_entity_ids')
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

    print(f"Positive pairs (true matches): {df_pairs['is_true_match'].sum()}")
    print(f"Negative pairs (non-matches): {(df_pairs['is_true_match'] == 0).sum()}")

    # =================== FEATURE ENGINEERING ===================
    print("\n========== STAGE 2: FEATURE ENGINEERING ==========")
    df_features = build_features_for_pairs(df_pairs, df_s1, df_pool, blocker=blocker)

    # =================== TRAIN XGBOOST ===================
    models, best_thresh, feature_cols = train_xgboost(
        df_features, df_features['is_true_match'], df_features['source1_entity_id'], 
        args.output_dir, gt
    )

    # =================== LLM SNIPER ON BORDERLINE PAIRS ===================
    if not args.skip_llm:
        print("\n========== LLM SNIPER (Borderline Arbitration) ==========")
        X_all = df_features[feature_cols]
        all_proba = models[-1].predict_proba(X_all)[:, 1]
        df_features['xgb_prob'] = all_proba
        
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
                df_features.loc[borderline_mask, 'final_decision'] = llm_decisions
                print(f"LLM flipped {sum(1 for d in llm_decisions if d == 1)} borderline pairs to MATCH.")
            except Exception as e:
                print(f"LLM Sniper failed: {e}. Using XGBoost threshold only.")
        else:
            print(f"Skipping LLM (borderline pairs: {n_borderline})")

    print("\n========== TRAINING PIPELINE COMPLETE ==========")

if __name__ == "__main__":
    main()
