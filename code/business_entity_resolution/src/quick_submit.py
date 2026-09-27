"""Deadline fallback: exact-name + pincode matches, no training or large model.

Runs on the COMPLETE test split in bounded chunks. Both TSVs contain every S1
ID; matches are a subset of candidate pairs. Accuracy is deliberately limited.
"""
import argparse
import os
import sqlite3

import pandas as pd

from data_cleaner import clean_text, standardize_business_name, extract_pincode


def rows(path, chunksize):
    for chunk in pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, chunksize=chunksize,
                             usecols=['entity_id', 'business_name', 'business_address']):
        yield chunk


def key(name, address):
    name = standardize_business_name(clean_text(name))
    pin = extract_pincode(clean_text(address))
    return (name, pin) if name and pin else None


def build(test_dir, output_dir, chunksize=20000):
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, 'quick_exact.sqlite')
    # Replace a previous interrupted run; write TSVs to temporary paths and
    # publish only after both are complete.
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    db.execute('CREATE TABLE pool (name TEXT, pin TEXT, entity_id TEXT)')
    db.execute('CREATE INDEX pool_key ON pool (name, pin)')
    try:
        for source in (2, 3):
            path = os.path.join(test_dir, f'test_source{source}.tsv')
            for chunk in rows(path, chunksize):
                records = []
                for eid, name, address in chunk.itertuples(index=False, name=None):
                    k = key(name, address)
                    if k:
                        records.append((*k, eid))
                db.executemany('INSERT INTO pool VALUES (?,?,?)', records)
                db.commit()
            print(f'Indexed test source{source}', flush=True)
        matching_tmp = os.path.join(output_dir, 'matching_results.tsv.tmp')
        candidate_tmp = os.path.join(output_dir, 'candidate_pairs.tsv.tmp')
        with open(matching_tmp, 'w') as matching, open(candidate_tmp, 'w') as candidate:
            matching.write('source1_entity_id\tmatched_entity_ids\n')
            candidate.write('source1_entity_id\tcandidate_entity_ids\n')
            for chunk in rows(os.path.join(test_dir, 'test_source1.tsv'), chunksize):
                for eid, name, address in chunk.itertuples(index=False, name=None):
                    k = key(name, address)
                    matches = sorted(set(r[0] for r in db.execute(
                        'SELECT entity_id FROM pool WHERE name=? AND pin=?', k))) if k else []
                    joined = ','.join(matches)
                    matching.write(f'{eid}\t{joined}\n')
                    candidate.write(f'{eid}\t{joined}\n')
        os.replace(matching_tmp, os.path.join(output_dir, 'matching_results.tsv'))
        os.replace(candidate_tmp, os.path.join(output_dir, 'candidate_pairs.tsv'))
        print(f'Wrote both TSVs to {output_dir}; run the official validator before submitting.', flush=True)
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--chunksize', type=int, default=20000)
    args = parser.parse_args()
    build(args.test_dir, args.output_dir, args.chunksize)
