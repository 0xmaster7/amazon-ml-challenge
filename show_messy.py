import pandas as pd

print("Loading ground truth to find a match...")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

# Find a row that has at least one match
gt_matches = gt[gt['matched_entity_ids'].notna()]
example_match = gt_matches.iloc[0]

s1_id = example_match['source1_entity_id']
matched_ids = example_match['matched_entity_ids'].split(',')

print(f"\nLooking for: {s1_id}")
print(f"Matches to find: {matched_ids}")

# Read chunks to find the exact rows without loading 1.5GB into RAM
def find_row(filename, target_id):
    for chunk in pd.read_csv(filename, sep="\t", chunksize=10000):
        match = chunk[chunk['entity_id'] == target_id]
        if not match.empty:
            return match.iloc[0]
    return None

print("\nScanning Source 1...")
s1_row = find_row("dataset/train/train_source1.tsv", s1_id)

print("\nScanning Source 2 and 3...")
matched_rows = []
for m_id in matched_ids:
    if m_id.startswith('S2'):
        row = find_row("dataset/train/train_source2.tsv", m_id)
        if row is not None: matched_rows.append(row)
    elif m_id.startswith('S3'):
        row = find_row("dataset/train/train_source3.tsv", m_id)
        if row is not None: matched_rows.append(row)

print("="*50)
print(f"SOURCE 1 (The Master Record):")
print(f"Name: {s1_row['business_name']}")
print(f"Addr: {s1_row['business_address']}")
print("-" * 50)

for i, row in enumerate(matched_rows):
    print(f"MATCH {i+1} (The Messy Record from {row['entity_id'][:2]}):")
    print(f"Name: {row['business_name']}")
    print(f"Addr: {row['business_address']}")
    print("-" * 50)
