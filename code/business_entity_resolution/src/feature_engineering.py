import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, distance
from data_cleaner import mem_rss

# Multilingual cross-encoder (14 languages incl. French/Hindi scripts).
# FIXED: was cross-encoder/ms-marco-MiniLM-L-6-v2 - an ENGLISH passage ranker.
# The test set contains France (unseen in training), so the pair scorer must
# be multilingual. Alternative: BAAI/bge-reranker-v2-m3 (heavier, stronger).
CROSS_ENCODER_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_website(text):
    # Common TLDs incl. .in/.fr - the original list missed Indian/French domains
    if not text: return 0
    return 1 if re.search(r'(www\.|https?://|\.(com|net|org|co|in|fr|io|biz|store|shop)\b)', str(text).lower()) else 0

def extract_domain_alpha(text):
    """Extracts alphabetic substring from a URL-as-name for comparison."""
    if not text: return ""
    m = re.search(r'(?:www\.|https?://)?([a-z0-9]+)', str(text).lower())
    return m.group(1) if m else str(text).lower()

def extract_numbers(text):
    if not text: return set()
    return set(re.findall(r'\d+', str(text)))

def get_numeric_match_ratio(str1, str2):
    nums1, nums2 = extract_numbers(str1), extract_numbers(str2)
    if not nums1 and not nums2: return 1.0
    if not nums1 or not nums2: return 0.0
    return len(nums1.intersection(nums2)) / max(len(nums1), len(nums2))

def get_trigram_cosine(s1, s2):
    if not s1 or not s2: return 0.0
    s1, s2 = str(s1), str(s2)
    def trigrams(s): return set(s[i:i+3] for i in range(len(s)-2))
    t1, t2 = trigrams(s1), trigrams(s2)
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / np.sqrt(len(t1) * len(t2))

def has_mojibake(text):
    if not text: return 0
    non_ascii = len(re.findall(r'[^\x00-\x7F]', str(text)))
    return 1 if non_ascii > (len(str(text)) * 0.3) else 0

def check_landmark(text):
    if not text: return 0
    return 1 if re.search(r'\b(near|opp|opposite|behind|beside)\b', str(text).lower()) else 0

def street_token_overlap(street1, street2):
    """Jaccard overlap on street tokens (house number stripped; city tokens remain)."""
    if not street1 or not street2: return 0.0
    t1 = set(str(street1).split())
    t2 = set(str(street2).split())
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

# ============================================================
# MAIN FEATURE BUILDER
# ============================================================

def _build_features_chunk(df_pairs, df_s1, df_pool, blocker=None, use_cross_encoder=True, cross_model=None):
    """
    Takes candidate pairs with blocking metadata, merges raw text,
    and computes the FULL feature list for XGBoost.

    use_cross_encoder: pass the SAME value in training and inference
    (--no_cross_encoder flag on both scripts) so the feature set is consistent.
    Cross-encoder failures now raise instead of silently zeroing the feature:
    a silent 0.0 in only one of train/inference is a hidden distribution shift.
    """
    print("Merging text for feature engineering...")

    df_s1 = df_s1.copy()
    df_pool = df_pool.copy()
    df_s1['raw_name'] = df_s1['business_name'].fillna("").str.lower()
    df_pool['raw_name'] = df_pool['business_name'].fillna("").str.lower()

    merge_cols_s1 = ['entity_id', 'clean_name', 'expanded_name', 'raw_name',
                     'clean_address', 'raw_address', 'country', 'country_norm',
                     'extracted_pin', 'house_number', 'street_tokens',
                     'city_tag', 'state_tag', 'phone_keys']
    merge_cols_pool = merge_cols_s1[:]

    df = df_pairs.merge(df_s1[merge_cols_s1], left_on='source1_entity_id', right_on='entity_id', how='left')
    df.rename(columns={c: c + '_s1' for c in merge_cols_s1 if c != 'entity_id'}, inplace=True)
    df.drop('entity_id', axis=1, inplace=True)

    df = df.merge(df_pool[merge_cols_pool], left_on='candidate_entity_id', right_on='entity_id', how='left')
    df.rename(columns={c: c + '_cand' for c in merge_cols_pool if c != 'entity_id'}, inplace=True)
    df.drop('entity_id', axis=1, inplace=True)
    for c in df.columns:
        if str(df[c].dtype) == 'category':
            df[c] = df[c].astype(object)
    df.fillna("", inplace=True)

    # ===========================================================
    # NAME SIMILARITY FEATURES
    # ===========================================================
    print("  Computing name similarity features...")

    df['name_exact_match_raw'] = (df['raw_name_s1'] == df['raw_name_cand']).astype(int)
    df['name_exact_match_clean'] = (df['clean_name_s1'] == df['clean_name_cand']).astype(int)

    df['name_jaro_winkler'] = df.apply(
        lambda r: distance.JaroWinkler.normalized_similarity(r['clean_name_s1'], r['clean_name_cand']), axis=1)

    df['name_token_sort_ratio'] = df.apply(
        lambda r: fuzz.token_sort_ratio(r['clean_name_s1'], r['clean_name_cand']) / 100.0, axis=1)

    df['name_token_set_ratio'] = df.apply(
        lambda r: fuzz.token_set_ratio(r['clean_name_s1'], r['clean_name_cand']) / 100.0, axis=1)

    df['name_trigram_cosine'] = df.apply(
        lambda r: get_trigram_cosine(r['clean_name_s1'], r['clean_name_cand']), axis=1)

    if blocker is not None and getattr(blocker, 's1_embeddings', None) is not None:
        print("  Computing embedding cosine similarity from Stage 1...")
        df['embedding_cosine_sim'] = df.apply(
            lambda r: blocker.get_embedding_cosine_sim(r['source1_entity_id'], r['candidate_entity_id']), axis=1)
    elif 'faiss_score' in df.columns:
        # Resumed from a checkpoint without the live blocker: the FAISS score
        # recorded during blocking IS the same cosine similarity (fp16 rounding
        # aside). Pairs not caught by Layer 3 carry 0.0, same as before.
        print("  Using faiss_score column as embedding cosine similarity (resumed run, no live blocker)...")
        df['embedding_cosine_sim'] = df['faiss_score'].astype(float)
    else:
        df['embedding_cosine_sim'] = 0.0

    # Website/URL flags + domain comparison
    df['is_website_s1'] = df['raw_name_s1'].apply(is_website)
    df['is_website_cand'] = df['raw_name_cand'].apply(is_website)

    # FIXED: extract_domain_alpha was defined but never used - the
    # website-as-name case from the strategy doc had no bridging feature.
    # Compare a URL-name's domain token against the other side's clean name.
    df['domain_alpha_s1'] = df['raw_name_s1'].apply(lambda t: extract_domain_alpha(t) if is_website(t) else "")
    df['domain_alpha_cand'] = df['raw_name_cand'].apply(lambda t: extract_domain_alpha(t) if is_website(t) else "")

    def domain_name_sim(r):
        best = 0.0
        if r['domain_alpha_s1']:
            best = max(best, fuzz.partial_ratio(r['domain_alpha_s1'], r['clean_name_cand']) / 100.0)
        if r['domain_alpha_cand']:
            best = max(best, fuzz.partial_ratio(r['domain_alpha_cand'], r['clean_name_s1']) / 100.0)
        return best
    df['domain_name_sim'] = df.apply(domain_name_sim, axis=1)

    df['name_expanded_jw'] = df.apply(
        lambda r: distance.JaroWinkler.normalized_similarity(r['expanded_name_s1'], r['expanded_name_cand']), axis=1)

    # ===========================================================
    # ADDRESS SIMILARITY FEATURES
    # ===========================================================
    print("  Computing address similarity features...")

    df['address_missing'] = ((df['clean_address_s1'] == "") | (df['clean_address_cand'] == "")).astype(int)

    df['addr_fuzzy_ratio'] = df.apply(
        lambda r: fuzz.ratio(r['clean_address_s1'], r['clean_address_cand']) / 100.0
        if r['address_missing'] == 0 else 0.0, axis=1)

    df['house_number_match'] = ((df['house_number_s1'] != "") &
                                 (df['house_number_s1'] == df['house_number_cand'])).astype(int)

    df['pincode_exact_match'] = ((df['extracted_pin_s1'] != "") &
                                  (df['extracted_pin_s1'] == df['extracted_pin_cand'])).astype(int)

    df['street_token_overlap'] = df.apply(
        lambda r: street_token_overlap(r['street_tokens_s1'], r['street_tokens_cand']), axis=1)

    # FIXED: landmark flags were computed on clean_address AFTER cleaning had
    # already deleted landmark words, so they were always 0. Now computed on
    # the raw lowercased address.
    df['has_landmark_s1'] = df['raw_address_s1'].apply(check_landmark)
    df['has_landmark_cand'] = df['raw_address_cand'].apply(check_landmark)

    df['addr_numeric_match_ratio'] = df.apply(
        lambda r: get_numeric_match_ratio(r['clean_address_s1'], r['clean_address_cand']), axis=1)

    # ===========================================================
    # STRUCTURAL / METADATA FEATURES
    # ===========================================================
    print("  Computing structural features...")

    # FIXED: country equality now uses normalized labels ("US" vs "USA" no
    # longer breaks the match). Still simple equality, NOT one-hot - France
    # appears only in test and must flow through as a label.
    df['country_match'] = (df['country_norm_s1'] == df['country_norm_cand']).astype(int)

    # City match: 1 same city, -1 both detected and different, 0 unknown
    # (unknown must NOT look like a mismatch - strategy doc: missing values
    # should be flags/neutral, never silent zeros).
    def _city_match(a, b):
        if not a or not b:
            return 0
        return 1 if a == b else -1
    df['city_match'] = [_city_match(a, b) for a, b in zip(df['city_tag_s1'], df['city_tag_cand'])]

    def _state_match(a, b):
        if not a or not b:
            return 0
        return 1 if a == b else -1
    df['state_match'] = [_state_match(a, b) for a, b in zip(df['state_tag_s1'], df['state_tag_cand'])]

    # Shared phone/tax-ID key (7+ digit run) - very strong signal when present
    def _pk(v):
        if isinstance(v, str):
            return set(v.split())
        return set(v) if isinstance(v, list) else set()
    df['phone_key_match'] = [1 if _pk(a) & _pk(b) else 0
                             for a, b in zip(df['phone_keys_s1'], df['phone_keys_cand'])]

    df['is_s3_candidate'] = df['candidate_entity_id'].astype(str).str.startswith('S3').astype(int)
    df['name_length_diff'] = abs(df['clean_name_s1'].str.len() - df['clean_name_cand'].str.len())
    df['is_mojibake_s1'] = df['raw_name_s1'].apply(has_mojibake)
    df['is_mojibake_cand'] = df['raw_name_cand'].apply(has_mojibake)

    # ===========================================================
    # CROSS-ENCODER SCORE
    # ===========================================================
    if use_cross_encoder:
        print(f"  Computing Cross-Encoder scores ({CROSS_ENCODER_MODEL})...")
        try:
            if cross_model is None:
                from sentence_transformers import CrossEncoder
                cross_model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=128)
            pairs = list(zip(
                (df['clean_name_s1'] + " " + df['clean_address_s1']).tolist(),
                (df['clean_name_cand'] + " " + df['clean_address_cand']).tolist()
            ))
            raw_scores = cross_model.predict(pairs, show_progress_bar=True, batch_size=256)
            # ms-marco/mmarco cross-encoders emit raw logits; squash to (0,1)
            df['cross_encoder_score'] = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores)))
        except Exception as e:
            # FIXED: was a silent 0.0 fill. If this fails in only one of
            # train/inference you get a hidden distribution shift. Fail loudly
            # instead; use --no_cross_encoder on BOTH scripts to disable.
            raise RuntimeError(
                f"Cross-encoder failed: {e}. Either fix the model load or run "
                f"BOTH train.py and predict.py with --no_cross_encoder."
            )
    else:
        print("  Cross-encoder disabled (--no_cross_encoder).")

    # ===========================================================
    # DROP RAW STRING COLUMNS (keep only numeric features for XGBoost)
    # ===========================================================
    string_cols = [c for c in df.columns if any(c.endswith(s) for s in
                   ['_s1', '_cand']) and df[c].dtype == 'object']
    extra_drops = ['candidate_entity_ids', 'matched_entity_ids']
    for c in string_cols + extra_drops:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)

    print(f"  Feature Engineering Complete. Total features: {len([c for c in df.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'is_true_match']])}")
    return df


def build_features_for_pairs(df_pairs, df_s1, df_pool, blocker=None,
                             use_cross_encoder=True, chunk_size=200000):
    """Memory-bounded wrapper: features are built in chunks of pairs so the
    string-heavy merged frame never covers the whole candidate set at once
    (this was the next OOM after Layer 2 on a 13GB box). The cross-encoder
    model loads ONCE and is reused across chunks."""
    import gc
    n = len(df_pairs)
    if n <= chunk_size:
        return _build_features_chunk(df_pairs, df_s1, df_pool, blocker=blocker,
                                     use_cross_encoder=use_cross_encoder)
    n_chunks = (n + chunk_size - 1) // chunk_size
    print(f"Building features in {n_chunks} chunks of {chunk_size} pairs (memory-bounded)...")
    cross_model = None
    if use_cross_encoder:
        from sentence_transformers import CrossEncoder
        cross_model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=128)
    parts = []
    for ci, start in enumerate(range(0, n, chunk_size)):
        print(f"  Feature chunk {ci + 1}/{n_chunks}... [mem {mem_rss():.1f}GB]")
        part = _build_features_chunk(
            df_pairs.iloc[start:start + chunk_size].copy(), df_s1, df_pool,
            blocker=blocker, use_cross_encoder=use_cross_encoder,
            cross_model=cross_model)
        parts.append(part)
        del part
        gc.collect()
    out = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"  All chunks done. Total rows: {len(out)}")
    return out
