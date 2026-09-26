import pandas as pd
import numpy as np
import os
import gc
import json
import hashlib
import pickle
from data_cleaner import mem_rss

class LayeredBlocker:
    def __init__(self):
        # (s1_id, cand_id) -> {'layers': [...], 'faiss_rank': int, 'faiss_score': float}
        self.candidate_pairs = {}
        # Store embeddings so we can reuse cosine sim as an XGBoost feature
        self.s1_embeddings = None
        self.pool_embeddings = None
        self.pool_ids = None
        self.s1_ids = None

    def save_progress(self, ckpt_dir, done_layers):
        """Persist candidate pairs + completed layer names after each layer,
        so an OOM/session kill mid-blocking doesn't lose earlier layers."""
        path = os.path.join(ckpt_dir, "blocker_progress.pkl")
        with open(path, 'wb') as f:
            pickle.dump({'pairs': self.candidate_pairs, 'done': sorted(done_layers)}, f)
        print(f"[checkpoint] blocker progress saved ({len(self.candidate_pairs)} pairs, done: {sorted(done_layers)})")

    def load_progress(self, ckpt_dir):
        """Returns the set of completed layer names; restores candidate pairs."""
        path = os.path.join(ckpt_dir, "blocker_progress.pkl")
        if not os.path.exists(path):
            return set()
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.candidate_pairs = d['pairs']
        print(f"[checkpoint] blocker resumed: {len(self.candidate_pairs)} pairs, layers done: {d['done']}")
        return set(d['done'])

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
        df_pool = pd.concat([df_s2[['entity_id', 'name_first_token', 'extracted_pin']],
                             df_s3[['entity_id', 'name_first_token', 'extracted_pin']]])
        pool_valid = df_pool[(df_pool['extracted_pin'] != "") & (df_pool['name_first_token'] != "")]
        s1_valid = df_s1[(df_s1['extracted_pin'] != "") & (df_s1['name_first_token'] != "")]

        k = {'name_first_token': str, 'extracted_pin': str}
        merged = pd.merge(
            s1_valid[['entity_id', 'name_first_token', 'extracted_pin']].astype(k),
            pool_valid[['entity_id', 'name_first_token', 'extracted_pin']].astype(k),
            on=['name_first_token', 'extracted_pin'],
            suffixes=('_s1', '_cand')
        )
        for _, row in merged.iterrows():
            self._add_pair(row['entity_id_s1'], row['entity_id_cand'], 'layer1_exact')
        print(f"[mem {mem_rss():.1f}GB] Layer 1 complete. Total pairs so far: {len(self.candidate_pairs)}")

    def layer2_tfidf_blocking(self, df_s1, df_s2, df_s3,
                              sim_threshold=0.45, top_k=30,
                              pool_slice=100000, s1_batch=200000,
                              n_features=2 ** 24):
        """Typo safety net via TF-IDF over character n-grams, CONSTANT-memory.

        v2.5 rewrite after three OOM kills at this layer on a 13GB Kaggle box.
        The earlier versions' real peaks were structural, not tuning:
          1. pd.concat([df_s2, df_s3]) copied ALL ~19 object columns of the
             10M-row pool - a full second copy of the cleaned frames.
          2. pool_texts/s1_texts lists were a THIRD copy of all the text.
          3. The whole S1 TF-IDF matrix Q lived in RAM.
        Now: only the 3 needed columns are concatenated, text slices are built
        on demand, and Q is built in chunks saved to disk and loaded one chunk
        at a time during matching. HashingVectorizer (no vocab dict), 2^24
        buckets (collision noise negligible), sub-threshold entries compacted
        before any COO materialization. Same cosine math - recall unchanged.
        """
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.preprocessing import normalize
        import scipy.sparse as sp

        print("Running Layer 2: TF-IDF char-n-gram blocking (constant-memory)...")
        df_pool = pd.concat([
            df_s2[['entity_id', 'clean_name', 'clean_address']],
            df_s3[['entity_id', 'clean_name', 'clean_address']],
        ]).drop_duplicates('entity_id').reset_index(drop=True)
        pool_ids = df_pool['entity_id'].values
        s1_ids_arr = df_s1['entity_id'].values
        n_pool = len(df_pool)
        n_s1 = len(df_s1)
        N = n_pool + n_s1

        vec = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                n_features=n_features, norm=None,
                                alternate_sign=False, dtype=np.float32)

        def slice_texts(df, a, b):
            return (df['clean_name'].iloc[a:b] + " " + df['clean_address'].iloc[a:b]).tolist()

        # Pass 1: document frequency per bucket, slice by slice.
        print(f"  Pass 1/3: idf counts over {N} records in {pool_slice}-row slices...")
        df_counts = np.zeros(n_features, dtype=np.float64)
        for df_ in (df_pool, df_s1):
            for i in range(0, len(df_), pool_slice):
                X = vec.transform(slice_texts(df_, i, min(i + pool_slice, len(df_))))
                X.data[:] = 1.0  # binarize -> document frequency
                df_counts += np.asarray(X.sum(axis=0)).ravel()
                del X
                gc.collect()
        idf = np.log((1.0 + N) / (1.0 + df_counts)) + 1.0
        del df_counts
        gc.collect()

        # Pass 2: normalized TF-IDF for S1, in chunks ON DISK - Q never lives
        # whole in RAM no matter how big source1 is.
        scratch = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
        q_paths = []
        print(f"  Pass 2/3: vectorizing {n_s1} S1 records to disk chunks of {s1_batch}...")
        for ci, i in enumerate(range(0, n_s1, s1_batch)):
            X = vec.transform(slice_texts(df_s1, i, min(i + s1_batch, n_s1)))
            X = normalize(X.multiply(idf).tocsr(), norm='l2', copy=False)
            p = os.path.join(scratch, f"_l2_qchunk_{ci}.npz")
            sp.save_npz(p, X)
            q_paths.append(p)
            del X
            gc.collect()

        # Pass 3: match pool slice by slice; only >=threshold pairs ever
        # become Python objects.
        print(f"  Pass 3/3: matching {n_pool} pool records in {pool_slice}-row slices...")
        hits = {}  # s1_idx -> {pool_idx: best score}
        n_slices = (n_pool + pool_slice - 1) // pool_slice
        for p_start in range(0, n_pool, pool_slice):
            p_end = min(p_start + pool_slice, n_pool)
            print(f"  Pool slice [{p_start}:{p_end}] ({p_start // pool_slice + 1}/{n_slices})...")
            Ps = normalize(
                vec.transform(slice_texts(df_pool, p_start, p_end)).multiply(idf).tocsr(),
                norm='l2', copy=False)
            for ci, qp in enumerate(q_paths):
                Qc = sp.load_npz(qp)
                S = (Qc @ Ps.T).tocsr()
                S.data[S.data < sim_threshold] = 0.0
                S.eliminate_zeros()  # compact BEFORE materializing as COO
                S = S.tocoo()
                base = ci * s1_batch
                for i, j, v in zip(S.row, S.col, S.data):
                    gi, gj = base + int(i), p_start + int(j)
                    row = hits.get(gi)
                    if row is None:
                        hits[gi] = {gj: float(v)}
                    elif v > row.get(gj, -1.0):
                        row[gj] = float(v)
                del Qc, S
            del Ps
            gc.collect()
            # Bound accumulation: prune any row past 2x top_k.
            for k, row in hits.items():
                if len(row) > top_k * 2:
                    hits[k] = dict(sorted(row.items(), key=lambda t: -t[1])[:top_k])

        for qp in q_paths:
            os.remove(qp)
        del q_paths
        gc.collect()

        n_added = 0
        for s_idx, row in hits.items():
            s1_id = s1_ids_arr[s_idx]
            for pidx, score in sorted(row.items(), key=lambda t: -t[1])[:top_k]:
                self._add_pair(s1_id, pool_ids[pidx], 'layer2_tfidf')
                n_added += 1
        print(f"[mem {mem_rss():.1f}GB] Layer 2 complete. Added {n_added} pairs. Total pairs so far: {len(self.candidate_pairs)}")

    def _embed_cache_key(self, model_name, ids, texts):
        """Content-derived cache key for the memmap embedding cache.

        FIXED: the old cache was validated only by byte size, so same-row-count
        but different data (train vs test, or any cleaning change) silently
        reused the WRONG embeddings. This key covers model, row count, a text
        sample, and total character count.
        """
        h = hashlib.md5()
        h.update(model_name.encode())
        h.update(str(len(texts)).encode())
        n = len(texts)
        sample_idx = list(range(min(1000, n))) + list(range(max(0, n - 1000), n))
        for i in sample_idx:
            h.update(str(ids[i]).encode())
            h.update(b'\x00')
            h.update(texts[i].encode('utf-8', errors='ignore'))
            h.update(b'\x00')
        h.update(str(sum(len(t) for t in texts)).encode())
        return h.hexdigest()

    def _encode_to_memmap(self, model, texts, mmap_path, d, chunk_size=500000):
        print(f"Encoding {len(texts)} texts in chunks with memmap -> {mmap_path}...")
        if os.path.exists(mmap_path):
            os.remove(mmap_path)
        emb_mm = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(len(texts), d))
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            print(f"  Encoding chunk {i} to {i + len(chunk)}...")
            emb_chunk = model.encode(chunk, show_progress_bar=True,
                                     normalize_embeddings=True, batch_size=1024)
            emb_mm[i:i + len(chunk)] = emb_chunk
            emb_mm.flush()
            del emb_chunk, chunk
            gc.collect()
        del emb_mm
        gc.collect()

    def layer3_semantic_embeddings(self, df_s1, df_s2, df_s3, model_name=None):
        # Lazy imports so layers 1/2/4 run without the heavy ML stack installed
        import torch
        import faiss
        from sentence_transformers import SentenceTransformer
        if model_name is None:
            if os.path.exists("../../output/finetuned_embedder"):
                model_name = "../../output/finetuned_embedder"
                print(f"Using fine-tuned embedder: {model_name}")
            else:
                model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

        print(f"Running Layer 3: FAISS Semantic Embeddings ({model_name})...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print(f"Skipping Layer 3: {e}")
            return

        df_pool = pd.concat([df_s2[['entity_id', 'embed_text']],
                             df_s3[['entity_id', 'embed_text']]]).reset_index(drop=True)
        # FIXED: embed the raw-preserved text (scripts/accents intact), not the
        # ASCII-stripped clean text - the multilingual model needs multilingual input.
        pool_texts = df_pool['embed_text'].tolist()
        s1_texts = df_s1['embed_text'].tolist()
        pool_ids = df_pool['entity_id'].values
        s1_ids = df_s1['entity_id'].values

        d = model.get_sentence_embedding_dimension()

        import re as _re
        mmap_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
        # Cache files are per-model: a second embedder (ensemble) or a
        # fine-tuned checkpoint gets its own memmaps instead of clobbering.
        model_slug = _re.sub(r'[^A-Za-z0-9]+', '_', model_name)[-50:]
        pool_mmap_path = os.path.join(mmap_dir, f"pool_embeddings_{model_slug}.dat")
        s1_mmap_path = os.path.join(mmap_dir, f"s1_embeddings_{model_slug}.dat")

        # Content-keyed cache validation (see _embed_cache_key)
        for mmap_path, ids, texts, tag in (
            (pool_mmap_path, pool_ids, pool_texts, "pool"),
            (s1_mmap_path, s1_ids, s1_texts, "s1"),
        ):
            key = self._embed_cache_key(model_name, ids, texts)
            key_path = mmap_path + ".key"
            expected_bytes = len(texts) * d * 4
            cache_ok = (
                os.path.exists(mmap_path)
                and os.path.getsize(mmap_path) == expected_bytes
                and os.path.exists(key_path)
                and open(key_path).read().strip() == key
            )
            if cache_ok:
                print(f"Found VALID cached {mmap_path}. Skipping {tag} encoding!")
            else:
                if os.path.exists(mmap_path):
                    print(f"Cache key mismatch for {tag} embeddings - re-encoding (stale cache discarded).")
                self._encode_to_memmap(model, texts, mmap_path, d)
                with open(key_path, "w") as f:
                    f.write(key)

        self.pool_ids = pool_ids
        self.s1_ids = s1_ids
        n_pool = len(self.pool_ids)
        n_s1 = len(self.s1_ids)

        del pool_texts, s1_texts, df_pool
        gc.collect()

        self.pool_embeddings = np.memmap(pool_mmap_path, dtype='float32', mode='r', shape=(n_pool, d))
        self.s1_embeddings = np.memmap(s1_mmap_path, dtype='float32', mode='r', shape=(n_s1, d))
        self.pool_id_to_idx = {pid: i for i, pid in enumerate(self.pool_ids)}
        self.s1_id_to_idx = {sid: i for i, sid in enumerate(self.s1_ids)}

        # RECALL TUNING: candidate_pairs.tsv is not scored, so over-generation
        # is nearly free - the metric cost of a missed match (recall ceiling)
        # far exceeds Stage 2's filtering cost. top_k raised 20 -> 30,
        # threshold lowered 0.55 -> 0.50.
        top_k = 30
        sim_threshold = 0.50
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Searching embeddings with accelerator: {device}...")

        if device.type == 'cuda':
            pool_slice_size = 400000
            s1_batch_size = 1000
            n_pool_slices = (n_pool + pool_slice_size - 1) // pool_slice_size

            for p_idx in range(n_pool_slices):
                p_start = p_idx * pool_slice_size
                p_end = min(p_start + pool_slice_size, n_pool)
                slice_len = p_end - p_start
                print(f"  GPU Search: Pool slice {p_idx+1}/{n_pool_slices} [{p_start}:{p_end}]...")

                pool_tensor = torch.from_numpy(self.pool_embeddings[p_start:p_end]).to(device=device, dtype=torch.float16)

                n_batches = (n_s1 + s1_batch_size - 1) // s1_batch_size
                for b_idx in range(n_batches):
                    s_start = b_idx * s1_batch_size
                    s_end = min(s_start + s1_batch_size, n_s1)

                    q_tensor = torch.from_numpy(self.s1_embeddings[s_start:s_end]).to(device=device, dtype=torch.float16)
                    sim_matrix = torch.matmul(q_tensor, pool_tensor.T)

                    actual_k = min(top_k, slice_len)
                    scores, indices = torch.topk(sim_matrix, k=actual_k, dim=1)

                    scores_np = scores.cpu().numpy()
                    indices_np = indices.cpu().numpy()

                    s1_batch_ids = self.s1_ids[s_start:s_end]
                    for q_i, s1_id in enumerate(s1_batch_ids):
                        for rank in range(actual_k):
                            score = float(scores_np[q_i, rank])
                            if score > sim_threshold:
                                match_idx = int(indices_np[q_i, rank])
                                cand_id = self.pool_ids[p_start + match_idx]
                                self._add_pair(s1_id, cand_id, 'layer3_faiss', faiss_rank=rank, faiss_score=score)

                    del q_tensor, sim_matrix, scores, indices

                del pool_tensor
                torch.cuda.empty_cache()
                print(f"    Slice {p_idx+1}/{n_pool_slices} complete. Total candidate pairs: {len(self.candidate_pairs)}")
        else:
            print("  Falling back to FAISS CPU...")
            pool_slice_size = 2500000
            s1_search_batch = 50000
            for p_start in range(0, n_pool, pool_slice_size):
                p_end = min(p_start + pool_slice_size, n_pool)
                sub_index = faiss.IndexFlatIP(d)
                sub_index.add(np.array(self.pool_embeddings[p_start:p_end]).astype('float32'))
                for s_start in range(0, n_s1, s1_search_batch):
                    s_end = min(s_start + s1_search_batch, n_s1)
                    q_chunk = np.array(self.s1_embeddings[s_start:s_end]).astype('float32')
                    D_chunk, I_chunk = sub_index.search(q_chunk, k=top_k)
                    for idx, s1_id in enumerate(self.s1_ids[s_start:s_end]):
                        for j in range(top_k):
                            score = float(D_chunk[idx][j])
                            match_idx = I_chunk[idx][j]
                            if match_idx >= 0 and score > sim_threshold:
                                cand_id = self.pool_ids[p_start + match_idx]
                                self._add_pair(s1_id, cand_id, 'layer3_faiss', faiss_rank=j, faiss_score=score)
                    del q_chunk, D_chunk, I_chunk
                del sub_index
                gc.collect()

        print(f"[mem {mem_rss():.1f}GB] Layer 3 complete. Total pairs so far: {len(self.candidate_pairs)}")

    def layer4_address_only(self, df_s1, df_s2, df_s3):
        print("Running Layer 4: Address-Only Fallback...")
        df_pool = pd.concat([df_s2[['entity_id', 'clean_address']],
                             df_s3[['entity_id', 'clean_address']]])
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
        print(f"[mem {mem_rss():.1f}GB] Layer 4 complete. Total pairs so far: {len(self.candidate_pairs)}")

    def layer4b_near_exact_address(self, df_s1, df_s2, df_s3, jaccard_threshold=0.85,
                                   max_block=2000):
        """Near-exact address matching: within each shared pincode block,
        token-Jaccard on street tokens. Catches single-token address typos
        that exact-equality Layer 4 misses (strategy doc: exact/NEAR-exact).
        Blocks bigger than max_block rows on either side are skipped (a pincode
        that common is a city, not a block)."""
        print("Running Layer 4b: Near-Exact Address (token-Jaccard within pincode blocks)...")

        def tokset(s):
            return set(str(s).split()) if s else set()

        pool = pd.concat([df_s2[['entity_id', 'extracted_pin', 'street_tokens']],
                          df_s3[['entity_id', 'extracted_pin', 'street_tokens']]])
        pool_by_pin = {}
        for pin, grp in pool[pool['extracted_pin'] != ""].groupby('extracted_pin', observed=True):
            pool_by_pin[pin] = [(r['entity_id'], tokset(r['street_tokens']))
                                for _, r in grp.iterrows()]

        n_added = 0
        for pin, grp in df_s1[df_s1['extracted_pin'] != ""].groupby('extracted_pin', observed=True):
            cands = pool_by_pin.get(pin)
            if not cands or len(cands) > max_block or len(grp) > max_block:
                continue
            for _, r in grp.iterrows():
                t1 = tokset(r['street_tokens'])
                if not t1:
                    continue
                for cid, t2 in cands:
                    if not t2:
                        continue
                    inter = len(t1 & t2)
                    if inter and inter / len(t1 | t2) >= jaccard_threshold:
                        self._add_pair(r['entity_id'], cid, 'layer4b_near_address')
                        n_added += 1
        print(f"[mem {mem_rss():.1f}GB] Layer 4b complete. Added {n_added} pairs. Total pairs so far: {len(self.candidate_pairs)}")

    def layer5_phone_key_blocking(self, df_s1, df_s2, df_s3):
        """Blocks on shared long digit runs (phone / tax-ID / registration
        numbers embedded in the record text). Strategy doc: Indian records
        sometimes tuck these into the address string; a shared 7+ digit
        number is a very strong candidate signal."""
        print("Running Layer 5: Phone/ID Key Blocking...")
        pool = pd.concat([df_s2[['entity_id', 'phone_keys']],
                          df_s3[['entity_id', 'phone_keys']]])

        def explode(df):
            rows = []
            for _, r in df.iterrows():
                keys = r['phone_keys']
                if isinstance(keys, str):
                    keys = keys.split()
                for k in (keys or []):
                    rows.append((k, r['entity_id']))
            return rows

        pool_idx = {}
        for k, eid in explode(pool):
            pool_idx.setdefault(k, []).append(eid)

        n_added = 0
        for _, r in df_s1.iterrows():
            for k in (r['phone_keys'] or []):
                for cid in pool_idx.get(k, []):
                    self._add_pair(r['entity_id'], cid, 'layer5_phonekey')
                    n_added += 1
        print(f"[mem {mem_rss():.1f}GB] Layer 5 complete. Added {n_added} pairs. Total pairs so far: {len(self.candidate_pairs)}")

    def get_embedding_cosine_sim(self, s1_id, cand_id):
        """Returns the precomputed cosine similarity between an S1 and candidate embedding."""
        if self.s1_embeddings is None or self.pool_embeddings is None:
            return 0.0
        try:
            s1_idx = self.s1_id_to_idx.get(s1_id)
            pool_idx = self.pool_id_to_idx.get(cand_id)
            if s1_idx is None or pool_idx is None:
                return 0.0
            return float(np.dot(self.s1_embeddings[s1_idx], self.pool_embeddings[pool_idx]))
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
                'found_in_layer2': 1 if 'layer2_tfidf' in meta['layers'] else 0,
                'found_in_layer3': 1 if 'layer3_faiss' in meta['layers'] else 0,
                'found_in_layer4': 1 if 'layer4_address' in meta['layers'] else 0,
                'found_in_layer4b': 1 if 'layer4b_near_address' in meta['layers'] else 0,
                'found_in_layer5': 1 if 'layer5_phonekey' in meta['layers'] else 0,
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
