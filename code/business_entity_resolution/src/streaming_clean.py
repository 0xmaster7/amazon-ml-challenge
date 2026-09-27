"""Disk-backed cleaning and narrow stage views for Kaggle's 13 GB host.

Never unpickle the legacy all-source cleaned checkpoint: its peak is unsafe.
The manifest is published last, so an interrupted conversion is rebuilt.
"""
import gc
import json
import os
import pickle
import shutil
import sqlite3
import zlib

import pandas as pd

from data_cleaner import process_dataframe, mem_rss

FEATURE_COLUMNS = ['entity_id', 'business_name', 'clean_name', 'expanded_name',
                   'clean_address', 'raw_address', 'country', 'country_norm',
                   'extracted_pin', 'house_number', 'street_tokens', 'city_tag',
                   'state_tag', 'phone_keys']
LAYER_COLUMNS = {
    'layer1': ['entity_id', 'name_first_token', 'extracted_pin'],
    'layer2': ['entity_id', 'clean_name', 'clean_address'],
    'layer3': ['entity_id', 'embed_text'],
    'layer4': ['entity_id', 'clean_address'],
    'layer4b': ['entity_id', 'extracted_pin', 'street_tokens'],
    'layer5': ['entity_id', 'phone_keys'],
}


class CleanStore:
    def __init__(self, ckpt_dir, input_dir, prefix, resume, chunk_size=50000, max_rows=None):
        self.root = os.path.join(ckpt_dir, 'cleaned_' + prefix + '_shards')
        self.db_path = os.path.join(self.root, 'features.sqlite')
        self.sources = {s: os.path.join(input_dir, f'{prefix}_source{s}.tsv') for s in (1, 2, 3)}
        signature = {str(s): [os.path.getsize(p), os.stat(p).st_mtime_ns, max_rows, 'v2']
                     for s, p in self.sources.items()}
        manifest = os.path.join(self.root, 'manifest.json')
        valid = False
        metadata = None
        if resume and os.path.isfile(manifest) and os.path.isfile(self.db_path):
            with open(manifest) as f:
                metadata = json.load(f)
            valid = (metadata.get('signature') == signature and
                     set(metadata.get('shards', {})) == {'1', '2', '3'} and
                     all(os.path.isfile(os.path.join(self.root, path))
                         for paths in metadata['shards'].values() for path in paths))
        if not valid:
            # Never load cleaned_train.pkl. A fresh chunked read costs time, not RAM.
            if os.path.exists(self.root):
                shutil.rmtree(self.root)
            os.makedirs(self.root)
            db = sqlite3.connect(self.db_path)
            db.execute('CREATE TABLE features (source INTEGER, entity_id TEXT, country_norm TEXT, payload BLOB)')
            
            shards = {}
            for s, source in self.sources.items():
                shards[str(s)] = []
                for ci, raw in enumerate(pd.read_csv(source, sep='\t', chunksize=chunk_size, nrows=max_rows)):
                    frame = process_dataframe(raw)
                    name = f's{s}_{ci:05d}.pkl'
                    with open(os.path.join(self.root, name), 'wb') as f:
                        pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
                    shards[str(s)].append(name)
                    view = frame[FEATURE_COLUMNS].astype(object).where(pd.notna(frame[FEATURE_COLUMNS]), '')
                    db.executemany('INSERT INTO features VALUES (?,?,?,?)',
                                   ((s, str(r[0]), str(r[7]), zlib.compress(
                                       json.dumps([str(v) for v in r[1:]], ensure_ascii=False).encode('utf-8'), 3))
                                    for r in view.itertuples(index=False, name=None)))
                    db.commit()
                    del view, frame, raw
                    gc.collect()
                    print(f'[mem {mem_rss():.1f}GB] cleaned source{s} shard {ci + 1}', flush=True)
            db.execute('CREATE INDEX features_id ON features (source, entity_id)')
            db.commit()
            db.close()
            with open(manifest + '.tmp', 'w') as f:
                json.dump({'signature': signature, 'shards': shards}, f)
            os.replace(manifest + '.tmp', manifest)
        else:
            shards = metadata['shards']
            print('[checkpoint] resumed disk-backed cleaned shards')
        self.shards = shards

    def frames(self, source, cols):
        """Yield narrow views, dropping each full shard before the next."""
        for name in self.shards[str(source)]:
            with open(os.path.join(self.root, name), 'rb') as f:
                frame = pickle.load(f)
            view = frame[cols].copy()
            del frame
            yield view
            del view

    def stage(self, layer):
        cols = LAYER_COLUMNS[layer]
        result = []
        for source in (1, 2, 3):
            parts = list(self.frames(source, cols))
            result.append(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols))
            del parts
            gc.collect()
        print(f'[mem {mem_rss():.1f}GB] loaded {layer} narrow frames', flush=True)
        return result

    def feature_rows(self, source, ids):
        """Fetch only IDs in the current candidate-pair chunk, preserving source order."""
        ids = list({str(v) for v in ids})
        rows = []
        db = sqlite3.connect(self.db_path)
        for group_start in range(0, len(ids), 800):
            group = ids[group_start:group_start + 800]
            if not group:
                continue
            q = ','.join('?' for _ in group)
            rows.extend(db.execute(f'SELECT entity_id, payload FROM features WHERE source=? AND entity_id IN ({q})',
                                   (source, *group)))
        db.close()
        # Source IDs are expected unique; the original pair merge also expanded duplicates.
        return pd.DataFrame(((eid, *json.loads(zlib.decompress(payload))) for eid, payload in rows),
                            columns=FEATURE_COLUMNS)

    def pair_frames(self, pairs):
        s1 = self.feature_rows(1, pairs['source1_entity_id'].unique())
        ids = pairs['candidate_entity_id'].unique()
        pool = pd.concat([self.feature_rows(2, ids), self.feature_rows(3, ids)], ignore_index=True)
        return s1, pool

    def country_map(self, source=1):
        db = sqlite3.connect(self.db_path)
        values = dict(db.execute('SELECT entity_id, country_norm FROM features WHERE source=?', (source,)))
        db.close()
        return values

    def all_ids(self, source):
        db = sqlite3.connect(self.db_path)
        values = set(row[0] for row in db.execute('SELECT entity_id FROM features WHERE source=?', (source,)))
        db.close()
        return values
