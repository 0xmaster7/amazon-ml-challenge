import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, distance
from sentence_transformers import CrossEncoder

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_website(text):
    if not text: return 0
    return 1 if re.search(r'(\.com|\.net|\.org|\.co\.|www\.)', str(text).lower()) else 0

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
    """Jaccard overlap on street tokens (house number and city already stripped)."""
    if not street1 or not street2: return 0.0
    t1 = set(str(street1).split())
    t2 = set(str(street2).split())
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

# ============================================================
# MAIN FEATURE BUILDER
# ============================================================

def build_features_for_pairs(df_pairs, df_s1, df_pool, blocker=None):
    """
    Takes candidate pairs with blocking metadata, merges raw text,
    and computes the FULL feature list for XGBoost.
    """
    print("Merging text for feature engineering...")

    # Keep raw names for specific features
    df_s1 = df_s1.copy()
    df_pool = df_pool.copy()
    df_s1['raw_name'] = df_s1['business_name'].fillna("").str.lower()
    df_pool['raw_name'] = df_pool['business_name'].fillna("").str.lower()

    merge_cols_s1 = ['entity_id', 'clean_name', 'expanded_name', 'raw_name',
                     'clean_address', 'country', 'extracted_pin',
                     'house_number', 'street_tokens']
    merge_cols_pool = merge_cols_s1[:]

    df = df_pairs.merge(df_s1[merge_cols_s1], left_on='source1_entity_id', right_on='entity_id', how='left')
    df.rename(columns={c: c + '_s1' for c in merge_cols_s1 if c != 'entity_id'}, inplace=True)
    df.drop('entity_id', axis=1, inplace=True)

    df = df.merge(df_pool[merge_cols_pool], left_on='candidate_entity_id', right_on='entity_id', how='left')
    df.rename(columns={c: c + '_cand' for c in merge_cols_pool if c != 'entity_id'}, inplace=True)
    df.drop('entity_id', axis=1, inplace=True)
    df.fillna("", inplace=True)

    # ===========================================================
    # NAME SIMILARITY FEATURES
    # ===========================================================
    print("  Computing name similarity features...")

    # 1. Exact match (raw, after lowercasing + whitespace norm)
    df['name_exact_match_raw'] = (df['raw_name_s1'] == df['raw_name_cand']).astype(int)

    # 2. Legal-suffix-stripped exact match
    df['name_exact_match_clean'] = (df['clean_name_s1'] == df['clean_name_cand']).astype(int)

    # 3. Jaro-Winkler on cleaned names
    df['name_jaro_winkler'] = df.apply(
        lambda r: distance.JaroWinkler.normalized_similarity(r['clean_name_s1'], r['clean_name_cand']), axis=1)

    # 4. Token-sort Levenshtein ratio
    df['name_token_sort_ratio'] = df.apply(
        lambda r: fuzz.token_sort_ratio(r['clean_name_s1'], r['clean_name_cand']) / 100.0, axis=1)

    # 5. Token-set overlap ratio (Jaccard on word sets)
    df['name_token_set_ratio'] = df.apply(
        lambda r: fuzz.token_set_ratio(r['clean_name_s1'], r['clean_name_cand']) / 100.0, axis=1)

    # 6. Character trigram cosine similarity
    df['name_trigram_cosine'] = df.apply(
        lambda r: get_trigram_cosine(r['clean_name_s1'], r['clean_name_cand']), axis=1)

    # 7. Embedding cosine similarity (reused from Stage 1 blocker)
    if blocker is not None:
        print("  Computing embedding cosine similarity from Stage 1...")
        df['embedding_cosine_sim'] = df.apply(
            lambda r: blocker.get_embedding_cosine_sim(r['source1_entity_id'], r['candidate_entity_id']), axis=1)
    else:
        df['embedding_cosine_sim'] = 0.0

    # 8. Cross-encoder score (computed below)

    # 9. Website/URL flags + domain comparison
    df['is_website_s1'] = df['raw_name_s1'].apply(is_website)
    df['is_website_cand'] = df['raw_name_cand'].apply(is_website)

    # 10. Abbreviation-expanded Jaro-Winkler
    df['name_expanded_jw'] = df.apply(
        lambda r: distance.JaroWinkler.normalized_similarity(r['expanded_name_s1'], r['expanded_name_cand']), axis=1)

    # ===========================================================
    # ADDRESS SIMILARITY FEATURES
    # ===========================================================
    print("  Computing address similarity features...")

    # 11. Address missing flag
    df['address_missing'] = ((df['clean_address_s1'] == "") | (df['clean_address_cand'] == "")).astype(int)

    # 12. Full-address fuzzy string similarity
    df['addr_fuzzy_ratio'] = df.apply(
        lambda r: fuzz.ratio(r['clean_address_s1'], r['clean_address_cand']) / 100.0
        if r['address_missing'] == 0 else 0.0, axis=1)

    # 13. House/building number exact match
    df['house_number_match'] = ((df['house_number_s1'] != "") &
                                 (df['house_number_s1'] == df['house_number_cand'])).astype(int)

    # 14. Pincode/ZIP exact match
    df['pincode_exact_match'] = ((df['extracted_pin_s1'] != "") &
                                  (df['extracted_pin_s1'] == df['extracted_pin_cand'])).astype(int)

    # 15. Street-name token overlap (ignoring house number and city)
    df['street_token_overlap'] = df.apply(
        lambda r: street_token_overlap(r['street_tokens_s1'], r['street_tokens_cand']), axis=1)

    # 16. Landmark-reference flag
    df['has_landmark_s1'] = df['clean_address_s1'].apply(check_landmark)
    df['has_landmark_cand'] = df['clean_address_cand'].apply(check_landmark)

    # 17. Numeric-token match ratio (all numbers in address)
    df['addr_numeric_match_ratio'] = df.apply(
        lambda r: get_numeric_match_ratio(r['clean_address_s1'], r['clean_address_cand']), axis=1)

    # ===========================================================
    # STRUCTURAL / METADATA FEATURES
    # ===========================================================
    print("  Computing structural features...")

    # 18. Country match (simple equality, NOT one-hot)
    df['country_match'] = (df['country_s1'] == df['country_cand']).astype(int)

    # 19. Source pair type
    df['is_s3_candidate'] = df['candidate_entity_id'].astype(str).str.startswith('S3').astype(int)

    # 20. Name length difference
    df['name_length_diff'] = abs(df['clean_name_s1'].str.len() - df['clean_name_cand'].str.len())

    # 21. Mojibake flag
    df['is_mojibake_s1'] = df['raw_name_s1'].apply(has_mojibake)
    df['is_mojibake_cand'] = df['raw_name_cand'].apply(has_mojibake)

    # ===========================================================
    # BLOCKING-STAGE METADATA (already in df_pairs from blocker)
    # ===========================================================
    # found_in_layer1, found_in_layer2, found_in_layer3, found_in_layer4,
    # total_layers_caught, faiss_rank, faiss_score
    # These are already present in df from the merge with df_pairs — no action needed.

    # ===========================================================
    # CROSS-ENCODER SCORE
    # ===========================================================
    print("  Computing Cross-Encoder scores...")
    try:
        cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=128)
        pairs = list(zip(
            (df['clean_name_s1'] + " " + df['clean_address_s1']).tolist(),
            (df['clean_name_cand'] + " " + df['clean_address_cand']).tolist()
        ))
        df['cross_encoder_score'] = cross_model.predict(pairs, show_progress_bar=True, batch_size=256)
    except Exception as e:
        print(f"  Cross-encoder failed: {e}")
        df['cross_encoder_score'] = 0.0

    # ===========================================================
    # DROP RAW STRING COLUMNS (keep only numeric features + country_s1 metadata)
    # ===========================================================
    # Preserve country_s1 for stratified thresholding / country-specific evaluation
    string_cols = [c for c in df.columns if any(c.endswith(s) for s in
                   ['_s1', '_cand']) and df[c].dtype == 'object' and c != 'country_s1']
    # Also drop any leftover merge artifacts
    extra_drops = ['candidate_entity_ids', 'matched_entity_ids']
    for c in string_cols + extra_drops:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)

    print(f"  Feature Engineering Complete. Total features: {len([c for c in df.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'is_true_match', 'country_s1']])}")
    return df
