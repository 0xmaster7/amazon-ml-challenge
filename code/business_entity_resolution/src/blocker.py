import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
import faiss

class LayeredBlocker:
    def __init__(self):
        self.candidate_pairs = set()
        
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
        
        before = len(self.candidate_pairs)
        for _, row in merged.iterrows():
            self.candidate_pairs.add((row['entity_id_s1'], row['entity_id_cand']))
        print(f"Layer 1 added {len(self.candidate_pairs) - before} candidate pairs.")

    def _get_minhash(self, text):
        m = MinHash(num_perm=128)
        # Character trigrams
        for i in range(len(text) - 2):
            m.update(text[i:i+3].encode('utf8'))
        return m

    def layer2_minhash_lsh(self, df_s1, df_s2, df_s3):
        print("Running Layer 2: MinHash LSH for typo tolerance...")
        lsh = MinHashLSH(threshold=0.7, num_perm=128)
        
        df_pool = pd.concat([df_s2, df_s3])
        df_pool['combined_text'] = df_pool['clean_name'] + " " + df_pool['clean_address']
        df_s1['combined_text'] = df_s1['clean_name'] + " " + df_s1['clean_address']
        
        print("Hashing candidate pool...")
        for _, row in df_pool.iterrows():
            if len(row['combined_text']) > 3:
                lsh.insert(row['entity_id'], self._get_minhash(row['combined_text']))
                
        print("Querying S1 against LSH...")
        before = len(self.candidate_pairs)
        for _, row in df_s1.iterrows():
            if len(row['combined_text']) > 3:
                m_query = self._get_minhash(row['combined_text'])
                result = lsh.query(m_query)
                for cand_id in result:
                    self.candidate_pairs.add((row['entity_id'], cand_id))
                    
        print(f"Layer 2 added {len(self.candidate_pairs) - before} NEW candidate pairs.")

    def layer3_semantic_embeddings(self, df_s1, df_s2, df_s3):
        print("Running Layer 3: FAISS Semantic Embeddings (paraphrase-multilingual)...")
        # Load small, fast multilingual model
        try:
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        except Exception:
            print("Skipping Layer 3: SentenceTransformer not installed or failed to download.")
            return

        df_pool = pd.concat([df_s2, df_s3])
        
        # We encode Name + Address for rich semantics
        pool_texts = (df_pool['clean_name'] + " " + df_pool['clean_address']).tolist()
        s1_texts = (df_s1['clean_name'] + " " + df_s1['clean_address']).tolist()
        
        print("Encoding pool...")
        pool_embeddings = model.encode(pool_texts, show_progress_bar=True, normalize_embeddings=True)
        print("Encoding S1...")
        s1_embeddings = model.encode(s1_texts, show_progress_bar=True, normalize_embeddings=True)
        
        # Build FAISS index (Inner Product for normalized vectors = Cosine Sim)
        d = pool_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(np.array(pool_embeddings).astype('float32'))
        
        print("Searching top 50 matches in FAISS...")
        # k=50 because we know some S1s have 7+ matches
        D, I = index.search(np.array(s1_embeddings).astype('float32'), k=50)
        
        pool_ids = df_pool['entity_id'].values
        s1_ids = df_s1['entity_id'].values
        
        before = len(self.candidate_pairs)
        for i, s1_id in enumerate(s1_ids):
            for j in range(50):
                # Keep candidates with cosine sim > 0.65 to prevent massive memory blowouts
                if D[i][j] > 0.65:
                    cand_id = pool_ids[I[i][j]]
                    self.candidate_pairs.add((s1_id, cand_id))
                    
        print(f"Layer 3 added {len(self.candidate_pairs) - before} NEW candidate pairs.")

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
        
        before = len(self.candidate_pairs)
        for _, row in merged.iterrows():
            self.candidate_pairs.add((row['entity_id_s1'], row['entity_id_cand']))
        print(f"Layer 4 added {len(self.candidate_pairs) - before} NEW candidate pairs.")

    def export_candidate_pairs(self, output_path):
        print(f"Exporting {len(self.candidate_pairs)} total candidate pairs to {output_path}...")
        df_pairs = pd.DataFrame(list(self.candidate_pairs), columns=['source1_entity_id', 'candidate_entity_id'])
        grouped = df_pairs.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
        grouped.rename(columns={'candidate_entity_id': 'candidate_entity_ids'}, inplace=True)
        grouped.to_csv(output_path, sep='\t', index=False)
        print("Export complete.")
        return df_pairs
