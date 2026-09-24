import pandas as pd
import os
import argparse
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from data_cleaner import process_dataframe
from blocker import LayeredBlocker
from feature_engineering import build_features_for_pairs
from llm_sniper import LLMSniper

def f_05_score(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def train_xgboost(df_features, labels, groups):
    print("\nTraining XGBoost Classifier...")
    features = [c for c in df_features.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'match_label', 'is_true_match']]
    X = df_features[features]
    y = labels
    
    gkf = GroupKFold(n_splits=5)
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"Training Fold {fold+1}/5...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, early_stopping_rounds=50, random_state=42, tree_method="hist", device="cuda")
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
        models.append(clf)
        
        preds_proba = clf.predict_proba(X_val)[:, 1]
        best_thresh, best_score = 0.5, 0
        for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
            preds = (preds_proba > thresh).astype(int)
            score = f_05_score(y_val, preds)
            if score > best_score:
                best_score = score
                best_thresh = thresh
        print(f"Fold {fold+1} Best F-0.5 Score: {best_score:.4f} at Threshold: {best_thresh}")
        
    return models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../../output")
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    nrows = 1000 if args.test_mode else None

    df_s1 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source1.tsv"), sep="\t", nrows=nrows))
    df_s2 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source2.tsv"), sep="\t", nrows=nrows))
    df_s3 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source3.tsv"), sep="\t", nrows=nrows))
    df_pool = pd.concat([df_s2, df_s3])
    gt = pd.read_csv(os.path.join(args.data_dir, "train_ground_truth.tsv"), sep="\t")

    blocker = LayeredBlocker()
    blocker.layer1_exact_key_blocking(df_s1, df_s2, df_s3)
    blocker.layer2_minhash_lsh(df_s1, df_s2, df_s3)
    blocker.layer3_semantic_embeddings(df_s1, df_s2, df_s3)
    blocker.layer4_address_only(df_s1, df_s2, df_s3)
    
    df_pairs = blocker.export_candidate_pairs(os.path.join(args.output_dir, "candidate_pairs.tsv"))

    gt_exploded = gt.assign(matched_entity_ids=gt['matched_entity_ids'].str.split(',')).explode('matched_entity_ids')
    gt_exploded['is_true_match'] = 1
    df_pairs = df_pairs.merge(gt_exploded, left_on=['source1_entity_id', 'candidate_entity_id'], right_on=['source1_entity_id', 'matched_entity_ids'], how='left')
    df_pairs['is_true_match'] = df_pairs['is_true_match'].fillna(0)

    df_features = build_features_for_pairs(df_pairs, df_s1, df_pool)
    models = train_xgboost(df_features, df_features['is_true_match'], df_features['source1_entity_id'])

    print("\nPIPELINE COMPLETE.")

if __name__ == "__main__":
    main()
