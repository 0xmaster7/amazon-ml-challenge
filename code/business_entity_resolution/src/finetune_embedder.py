"""
finetune_embedder.py — Contrastive Fine-Tuning for Domain-Specific Embeddings

Fine-tunes 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' directly
on ground-truth positive pairs using MultipleNegativesRankingLoss.

Why this matters:
Generic embedders are trained on general Wikipedia/web text.
Fine-tuning teaches the model business name abbreviations, mojibake, DBA aliases,
and regional Indian/US/French address patterns, producing vastly higher-quality
cosine similarity rankings and blocking recall.
"""
import os
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from data_cleaner import process_dataframe


def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentence embedder on business entity pairs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_model_dir", type=str, default="../../output/finetuned_embedder", help="Where to save fine-tuned model")
    parser.add_argument("--base_model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (1 epoch is usually sufficient)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for contrastive loss (larger = more in-batch negatives)")
    parser.add_argument("--max_pairs", type=int, default=100000, help="Max positive pairs to use for training")
    args = parser.parse_args()

    print("=" * 70)
    print("      FINE-TUNING SENTENCE EMBEDDER ON COMPETITION PAIRS")
    print("=" * 70)

    # 1. LOAD DATA
    print("\n[1/4] Loading and cleaning datasets...")
    df_s1 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source1.tsv"), sep='\t'))
    df_s2 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source2.tsv"), sep='\t'))
    df_s3 = process_dataframe(pd.read_csv(os.path.join(args.data_dir, "train_source3.tsv"), sep='\t'))
    df_pool = pd.concat([df_s2, df_s3]).reset_index(drop=True)
    gt = pd.read_csv(os.path.join(args.data_dir, "train_ground_truth.tsv"), sep='\t')

    # Build text lookups
    df_s1['full_text'] = (df_s1['clean_name'] + " " + df_s1['clean_address']).fillna("")
    df_pool['full_text'] = (df_pool['clean_name'] + " " + df_pool['clean_address']).fillna("")
    s1_text_map = df_s1.set_index('entity_id')['full_text'].to_dict()
    pool_text_map = df_pool.set_index('entity_id')['full_text'].to_dict()

    # 2. BUILD TRAINING EXAMPLES (POSITIVE PAIRS)
    print("\n[2/4] Constructing contrastive training pairs from ground truth...")
    singleton_mask = gt['matched_entity_ids'].isna() | (gt['matched_entity_ids'] == "")
    gt_matches = gt[~singleton_mask].copy()
    gt_exploded = gt_matches.assign(
        matched_entity_ids=gt_matches['matched_entity_ids'].str.split(',')
    ).explode('matched_entity_ids')

    train_examples = []
    for _, row in gt_exploded.iterrows():
        s1_id = row['source1_entity_id']
        cand_id = row['matched_entity_ids']
        text1 = s1_text_map.get(s1_id, "")
        text2 = pool_text_map.get(cand_id, "")
        if text1 and text2 and len(text1) > 3 and len(text2) > 3:
            train_examples.append(InputExample(texts=[text1, text2]))
            if len(train_examples) >= args.max_pairs:
                break

    print(f"Total positive training pairs created: {len(train_examples)}")
    if len(train_examples) == 0:
        print("Error: No training pairs created. Check data.")
        return

    # 3. INITIALIZE MODEL & LOSS
    print(f"\n[3/4] Initializing base model: {args.base_model}...")
    model = SentenceTransformer(args.base_model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    # MultipleNegativesRankingLoss treats all other items in batch as negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 4. TRAIN
    print(f"\n[4/4] Training for {args.epochs} epoch(s) with batch size {args.batch_size}...")
    os.makedirs(args.output_model_dir, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=True,
        output_path=args.output_model_dir
    )

    print("\n" + "=" * 70)
    print(f"Fine-tuning complete! Model saved to: {args.output_model_dir}")
    print("To use in train.py / predict.py, specify the fine-tuned model path in blocker.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()
