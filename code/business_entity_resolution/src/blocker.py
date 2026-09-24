import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
import faiss

class LayeredBlocker:
    def __init__(self):
        # (s1_id, cand_id) -> {'layers': [...], 'faiss_rank': int, 'faiss_score': float}
        self.candidate_pairs = {}
        # Store embeddings so we can reuse cosine sim as an XGBoost feature
        self.s1_embeddings = None
        self.pool_embeddings = None
        self.pool_ids = None
        self.s1_ids = None
        
    def _add_pair(self, s1_id, cand_id, source_layer, faiss_rank=None, faiss_score=None):
        pair = (s1_id, cand_id)
        if pair not in self.candidate_pairs:
            self.candidate_pairs[pair] = {'layers': [], 'faiss_rank': -1, 'faiss_score': 0.0}
        meta = self.candidate_pairs[pair]
        if source_layer not in meta['layers']:
            meta['layers'].append(source_layer)
        if faiss_rank is not None and (meta['faiss_rank'] == -1 or faiss_rank < meta['faiss_rank']):
            meta['faiss_rank'] = faiss_rank
            meta['faiss_score'] = faiss_score
            
    def layer1_exact_key_blocking(self, df_s1, df_s2, df_s3):
        print("Running Layer 1: Exact Key Blocking...")
        df_pool = pd.concat([df_s2, df_s3])
        pool_valid = df_pool[(df_pool['extracted_pin'] != "") & (df_pool['name_first_token'] != "")]
        s1_valid = df_s1[(df_s1['extracted_pin'] != "") & (df_s1['name_first_token'] != "")]
        
        merged = pd.merge(
            s1_valid[['entity_id', 'name_first_token', 'extracted_pin']],
            pool_valid[['entity_id', 'name_first_token', 'extracted_pin']],
            on=['name_first_token', 'extracted_pin'],
            suffixes=('_s1', '_cand')
        )
        for _, row in merged.iterrows():
            self._add_pair(row['entity_id_s1'], row['entity_id_cand'], 'layer1_exact')
        print(f"Layer 1 complete. Total pairs so far: {len(self.candidate_pairs)}")

    def _get_minhash(self, text):
        m = MinHash(num_perm=128)
        for i in range(len(text) - 2):
            m.update(text[i:i+3].encode('utf8'))
        return m

    def layer2_minhash_lsh(self, df_s1, df_s2, df_s3):
        print("Running Layer 2: MinHash LSH for typo tolerance...")
        lsh = MinHashLSH(threshold=0.5, num_perm=128)
        df_pool = pd.concat([df_s2, df_s3])
        df_pool['combined_text'] = df_pool['clean_name'] + " " + df_pool['clean_address']
        df_s1['combined_text'] = df_s1['clean_name'] + " " + df_s1['clean_address']
        
        inserted = set()
        for _, row in df_pool.iterrows():
            if len(row['combined_text']) > 3 and row['entity_id'] not in inserted:
                try:
                    lsh.insert(row['entity_id'], self._get_minhash(row['combined_text']))
                    inserted.add(row['entity_id'])
                except ValueError:
                    pass  # duplicate key
                
        for _, row in df_s1.iterrows():
            if len(row['combined_text']) > 3:
                m_query = self._get_minhash(row['combined_text'])
                for cand_id in lsh.query(m_query):
                    self._add_pair(row['entity_id'], cand_id, 'layer2_minhash')
        print(f"Layer 2 complete. Total pairs so far: {len(self.candidate_pairs)}")

    def layer3_semantic_embeddings(self, df_s1, df_s2, df_s3):
        print("Running Layer 3: FAISS Semantic Embeddings (paraphrase-multilingual)...")
        try:
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            print(f"Skipping Layer 3: {e}")
            return

        df_pool = pd.concat([df_s2, df_s3]).reset_index(drop=True)
        pool_texts = (df_pool['clean_name'] + " " + df_pool['clean_address']).tolist()
        s1_texts = (df_s1['clean_name'] + " " + df_s1['clean_address']).tolist()
        
        print("Encoding pool...")
        pool_embeddings = model.encode(pool_texts, show_progress_bar=True, normalize_embeddings=True, batch_size=256)
        print("Encoding S1...")
        s1_embeddings = model.encode(s1_texts, show_progress_bar=True, normalize_embeddings=True, batch_size=256)
        
        # Store for reuse as XGBoost feature
        self.pool_embeddings = pool_embeddings
        self.s1_embeddings = s1_embeddings
        self.pool_ids = df_pool['entity_id'].values
        self.s1_ids = df_s1['entity_id'].values
        
        d = pool_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(np.array(pool_embeddings).astype('float32'))
        
        # k=50 because max fan-out in ground truth is 7, need big safety margin
        D, I = index.search(np.array(s1_embeddings).astype('float32'), k=50)
        
        for i, s1_id in enumerate(self.s1_ids):
            for j in range(50):
                if I[i][j] >= 0 and D[i][j] > 0.5:  # Lowered threshold for recall
                    cand_id = self.pool_ids[I[i][j]]
                    self._add_pair(s1_id, cand_id, 'layer3_faiss', faiss_rank=j, faiss_score=float(D[i][j]))
        print(f"Layer 3 complete. Total pairs so far: {len(self.candidate_pairs)}")

    def layer4_address_only(self, df_s1, df_s2, df_s3):
        print("Running Layer 4: Address-Only Fallback...")
        df_pool = pd.concat([df_s2, df_s3])
        pool_valid = df_pool[(df_pool['clean_address'] != "") & (df_pool['clean_address'].str.len() > 5)]
        s1_valid = df_s1[(df_s1['clean_address'] != "") & (df_s1['clean_address'].str.len() > 5)]
        
        merged = pd.merge(
            s1_valid[['entity_id', 'clean_address']],
            pool_valid[['entity_id', 'clean_address']],
            on='clean_address',
            suffixes=('_s1', '_cand')
        )
        for _, row in merged.iterrows():
            self._add_pair(row['entity_id_s1'], row['entity_id_cand'], 'layer4_address')
        print(f"Layer 4 complete. Total pairs so far: {len(self.candidate_pairs)}")

    def get_embedding_cosine_sim(self, s1_id, cand_id):
        """Returns the precomputed cosine similarity between an S1 and candidate embedding."""
        if self.s1_embeddings is None or self.pool_embeddings is None:
            return 0.0
        try:
            s1_idx = np.where(self.s1_ids == s1_id)[0]
            pool_idx = np.where(self.pool_ids == cand_id)[0]
            if len(s1_idx) == 0 or len(pool_idx) == 0:
                return 0.0
            return float(np.dot(self.s1_embeddings[s1_idx[0]], self.pool_embeddings[pool_idx[0]]))
        except Exception:
            return 0.0

    def export_candidate_pairs(self, output_path):
        print(f"Exporting {len(self.candidate_pairs)} candidates to {output_path}...")
        records = []
        for (s1, cand), meta in self.candidate_pairs.items():
            records.append({
                'source1_entity_id': s1,
                'candidate_entity_id': cand,
                'found_in_layer1': 1 if 'layer1_exact' in meta['layers'] else 0,
                'found_in_layer2': 1 if 'layer2_minhash' in meta['layers'] else 0,
                'found_in_layer3': 1 if 'layer3_faiss' in meta['layers'] else 0,
                'found_in_layer4': 1 if 'layer4_address' in meta['layers'] else 0,
                'total_layers_caught': len(meta['layers']),
                'faiss_rank': meta['faiss_rank'],
                'faiss_score': meta['faiss_score'],
            })
            
        df_pairs = pd.DataFrame(records)
        if df_pairs.empty:
            df_pairs = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id'])
            
        # Standard TSV export for leaderboard
        grouped = df_pairs.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
        grouped.rename(columns={'candidate_entity_id': 'candidate_entity_ids'}, inplace=True)
        grouped.to_csv(output_path, sep='\t', index=False)
        
        return df_pairs
