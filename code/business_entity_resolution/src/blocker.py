import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
import faiss

class LayeredBlocker:
    def __init__(self):
        # Dictionary mapping (s1_id, cand_id) -> list of strings representing which layers caught them
        self.candidate_pairs = {}
        
    def _add_pair(self, s1_id, cand_id, source_layer):
        pair = (s1_id, cand_id)
        if pair not in self.candidate_pairs:
            self.candidate_pairs[pair] = []
        if source_layer not in self.candidate_pairs[pair]:
            self.candidate_pairs[pair].append(source_layer)
            
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
        print("Layer 1 complete.")

    def _get_minhash(self, text):
        m = MinHash(num_perm=128)
        for i in range(len(text) - 2):
            m.update(text[i:i+3].encode('utf8'))
        return m

    def layer2_minhash_lsh(self, df_s1, df_s2, df_s3):
        print("Running Layer 2: MinHash LSH for typo tolerance...")
        lsh = MinHashLSH(threshold=0.7, num_perm=128)
        df_pool = pd.concat([df_s2, df_s3])
        df_pool['combined_text'] = df_pool['clean_name'] + " " + df_pool['clean_address']
        df_s1['combined_text'] = df_s1['clean_name'] + " " + df_s1['clean_address']
        
        for _, row in df_pool.iterrows():
            if len(row['combined_text']) > 3:
                lsh.insert(row['entity_id'], self._get_minhash(row['combined_text']))
                
        for _, row in df_s1.iterrows():
            if len(row['combined_text']) > 3:
                m_query = self._get_minhash(row['combined_text'])
                for cand_id in lsh.query(m_query):
                    self._add_pair(row['entity_id'], cand_id, 'layer2_minhash')
        print("Layer 2 complete.")

    def layer3_semantic_embeddings(self, df_s1, df_s2, df_s3):
        print("Running Layer 3: FAISS Semantic Embeddings...")
        try:
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        except Exception:
            print("Skipping Layer 3 (Not installed).")
            return

        df_pool = pd.concat([df_s2, df_s3])
        pool_texts = (df_pool['clean_name'] + " " + df_pool['clean_address']).tolist()
        s1_texts = (df_s1['clean_name'] + " " + df_s1['clean_address']).tolist()
        
        pool_embeddings = model.encode(pool_texts, show_progress_bar=True, normalize_embeddings=True)
        s1_embeddings = model.encode(s1_texts, show_progress_bar=True, normalize_embeddings=True)
        
        d = pool_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(np.array(pool_embeddings).astype('float32'))
        
        D, I = index.search(np.array(s1_embeddings).astype('float32'), k=50) # k=50 to catch 7+ matches
        
        pool_ids = df_pool['entity_id'].values
        s1_ids = df_s1['entity_id'].values
        
        for i, s1_id in enumerate(s1_ids):
            for j in range(50):
                if D[i][j] > 0.65:
                    self._add_pair(s1_id, pool_ids[I[i][j]], 'layer3_faiss')
        print("Layer 3 complete.")

    def layer4_address_only(self, df_s1, df_s2, df_s3):
        print("Running Layer 4: Address-Only Fallback...")
        df_pool = pd.concat([df_s2, df_s3])
        pool_valid = df_pool[df_pool['clean_address'] != ""]
        s1_valid = df_s1[df_s1['clean_address'] != ""]
        
        merged = pd.merge(
            s1_valid[['entity_id', 'clean_address']],
            pool_valid[['entity_id', 'clean_address']],
            on=['clean_address'],
            suffixes=('_s1', '_cand')
        )
        
        for _, row in merged.iterrows():
            self._add_pair(row['entity_id_s1'], row['entity_id_cand'], 'layer4_address')
        print("Layer 4 complete.")

    def export_candidate_pairs(self, output_path):
        print(f"Exporting candidates to {output_path}...")
        
        # Flatten dictionary to list of dicts to capture metadata
        records = []
        for (s1, cand), layers in self.candidate_pairs.items():
            records.append({
                'source1_entity_id': s1,
                'candidate_entity_id': cand,
                'found_in_layer1': 1 if 'layer1_exact' in layers else 0,
                'found_in_layer2': 1 if 'layer2_minhash' in layers else 0,
                'found_in_layer3': 1 if 'layer3_faiss' in layers else 0,
                'found_in_layer4': 1 if 'layer4_address' in layers else 0,
                'total_layers_caught': len(layers)
            })
            
        df_pairs = pd.DataFrame(records)
        if df_pairs.empty:
            df_pairs = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id'])
            
        # Standard Export for Leaderboard
        grouped = df_pairs.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
        grouped.rename(columns={'candidate_entity_id': 'candidate_entity_ids'}, inplace=True)
        grouped.to_csv(output_path, sep='\t', index=False)
        
        return df_pairs
