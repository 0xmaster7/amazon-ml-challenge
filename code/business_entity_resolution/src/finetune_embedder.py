"""
finetune_embedder.py — siamese/contrastive fine-tuning of the Stage 1 embedder
on YOUR training pairs (strategy doc point 3: usually the single biggest score
jump - it fixes retrieval AND the embedding-cosine feature at once).

Positive pairs come from train_ground_truth.tsv; MultipleNegativesRankingLoss
uses in-batch negatives, which are appropriately hard here because every
negative shares the same candidate pool.

    python finetune_embedder.py --data_dir <train dir> --output_dir ../../output

Saves to <output_dir>/finetuned_embedder - exactly where blocker.py's Layer 3
already looks for it, so train.py/predict.py pick it up automatically.
NOTE: delete output/checkpoints/ and old embedding memmaps after fine-tuning,
then re-run blocking - the cache key changes because the model path changes.
"""
import pandas as pd
import os
import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../../output")
    parser.add_argument("--base_model", type=str,
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_pairs", type=int, default=200000,
                        help="Cap training pairs; more is better but 100-200k is plenty.")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    print("Loading training data (RAW text - scripts and accents stay intact)...")
    s1 = pd.read_csv(os.path.join(args.data_dir, "train_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(args.data_dir, "train_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(args.data_dir, "train_source3.tsv"), sep="\t")
    gt = pd.read_csv(os.path.join(args.data_dir, "train_ground_truth.tsv"), sep="\t")

    def rowtext(df):
        return (df['business_name'].fillna("").astype(str) + " " +
                df['business_address'].fillna("").astype(str)).str.strip()

    text_by_id = {}
    for df in (s1, s2, s3):
        text_by_id.update(dict(zip(df['entity_id'], rowtext(df))))

    examples = []
    for _, row in gt.iterrows():
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or str(matched).strip() == '':
            continue
        anchor = text_by_id.get(row['source1_entity_id'], '')
        if not anchor:
            continue
        for m in str(matched).replace(' ', '').split(','):
            pos = text_by_id.get(m, '')
            if pos:
                examples.append(InputExample(texts=[anchor, pos]))
    random.seed(42)
    random.shuffle(examples)
    examples = examples[:args.max_pairs]
    print(f"Training on {len(examples)} positive pairs")

    model = SentenceTransformer(args.base_model)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    out_path = os.path.join(args.output_dir, "finetuned_embedder")
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
              warmup_steps=max(100, len(loader) // 10), output_path=out_path,
              show_progress_bar=True)
    print(f"Saved fine-tuned embedder to {out_path}")
    print("Layer 3 auto-detects ../../output/finetuned_embedder on the next run. "
          "Delete output/checkpoints/ first so blocking re-runs with the new space.")


if __name__ == "__main__":
    main()
