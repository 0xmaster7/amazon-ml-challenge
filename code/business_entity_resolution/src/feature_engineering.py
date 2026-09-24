import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, distance
from sentence_transformers import CrossEncoder

def is_website(text):
    if not text: return 0
    return 1 if re.search(r'(\.com|\.net|\.org|www\.)', str(text).lower()) else 0

def extract_numbers(text):
    if not text: return set()
    return set(re.findall(r'\d+', str(text)))

def get_numeric_match_ratio(str1, str2):
    nums1 = extract_numbers(str1)
    nums2 = extract_numbers(str2)
    if not nums1 and not nums2: return 1.0
    if not nums1 or not nums2: return 0.0
    return len(nums1.intersection(nums2)) / max(len(nums1), len(nums2))

def get_trigram_cosine(s1, s2):
    if not s1 or not s2: return 0.0
    s1, s2 = str(s1), str(s2)
    def get_trigrams(s): return set([s[i:i+3] for i in range(len(s)-2)])
    t1, t2 = get_trigrams(s1), get_trigrams(s2)
    if not t1 or not t2: return 0.0
    return len(t1.intersection(t2)) / np.sqrt(len(t1) * len(t2))

def has_mojibake(text):
    if not text: return 0
    # Simple heuristic for high-entropy / weird characters
    non_ascii = len(re.findall(r'[^\x00-\x7F]', str(text)))
    return 1 if non_ascii > (len(str(text)) * 0.3) else 0

def check_landmark(text):
    if not text: return 0
    return 1 if re.search(r'\b(near|opp|opposite|behind|beside)\b', str(text)) else 0

def build_features_for_pairs(df_pairs, df_s1, df_pool):
    print("Merging text for feature engineering...")
    
    # Keep the raw data for specific features
    df_s1['raw_name'] = df_s1['business_name'].fillna("").str.lower()
    df_pool['raw_name'] = df_pool['business_name'].fillna("").str.lower()

    df = df_pairs.merge(df_s1[['entity_id', 'clean_name', 'raw_name', 'clean_address', 'country', 'extracted_pin']], 
                        left_on='source1_entity_id', right_on='entity_id', how='left')
    df.rename(columns={'clean_name': 'name_s1', 'raw_name': 'raw_name_s1', 'clean_address': 'addr_s1', 'country': 'country_s1', 'extracted_pin': 'pin_s1'}, inplace=True)
    df.drop('entity_id', axis=1, inplace=True)
    
    df = df.merge(df_pool[['entity_id', 'clean_name', 'raw_name', 'clean_address', 'country', 'extracted_pin']], 
                  left_on='candidate_entity_id', right_on='entity_id', how='left')
    df.rename(columns={'clean_name': 'name_cand', 'raw_name': 'raw_name_cand', 'clean_address': 'addr_cand', 'country': 'country_cand', 'extracted_pin': 'pin_cand'}, inplace=True)
    df.drop('entity_id', axis=1, inplace=True)
    df.fillna("", inplace=True)

    print("Computing String Similarities...")
    df['name_exact_match_raw'] = (df['raw_name_s1'] == df['raw_name_cand']).astype(int)
    df['name_exact_match_clean'] = (df['name_s1'] == df['name_cand']).astype(int)
    
    df['name_jaro_winkler'] = df.apply(lambda row: distance.JaroWinkler.normalized_similarity(row['name_s1'], row['name_cand']), axis=1)
    df['name_token_sort_ratio'] = df.apply(lambda row: fuzz.token_sort_ratio(row['name_s1'], row['name_cand']) / 100.0, axis=1)
    df['name_token_set_ratio'] = df.apply(lambda row: fuzz.token_set_ratio(row['name_s1'], row['name_cand']) / 100.0, axis=1)
    df['name_trigram_cosine'] = df.apply(lambda row: get_trigram_cosine(row['name_s1'], row['name_cand']), axis=1)
    
    df['is_website_s1'] = df['raw_name_s1'].apply(is_website)
    df['is_website_cand'] = df['raw_name_cand'].apply(is_website)
    df['name_length_diff'] = abs(df['name_s1'].str.len() - df['name_cand'].str.len())
    
    df['is_mojibake'] = df['raw_name_s1'].apply(has_mojibake) | df['raw_name_cand'].apply(has_mojibake)
    
    df['address_missing'] = ((df['addr_s1'] == "") | (df['addr_cand'] == "")).astype(int)
    df['addr_fuzzy_ratio'] = df.apply(lambda row: fuzz.ratio(row['addr_s1'], row['addr_cand']) / 100.0 if row['address_missing'] == 0 else 0, axis=1)
    df['addr_numeric_match_ratio'] = df.apply(lambda row: get_numeric_match_ratio(row['addr_s1'], row['addr_cand']), axis=1)
    
    df['pincode_exact_match'] = ((df['pin_s1'] != "") & (df['pin_s1'] == df['pin_cand'])).astype(int)
    df['has_landmark'] = df['addr_s1'].apply(check_landmark) | df['addr_cand'].apply(check_landmark)
    
    df['country_match'] = (df['country_s1'] == df['country_cand']).astype(int)
    
    # Source pair type (S1-S2 vs S1-S3)
    df['is_s3_candidate'] = df['candidate_entity_id'].astype(str).str.contains('S3').astype(int)

    print("Computing Semantic Cross-Encoder Scores...")
    try:
        cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=128)
        pairs = list(zip(
            (df['name_s1'] + " " + df['addr_s1']).tolist(), 
            (df['name_cand'] + " " + df['addr_cand']).tolist()
        ))
        df['cross_encoder_score'] = cross_model.predict(pairs, show_progress_bar=True)
    except Exception as e:
        df['cross_encoder_score'] = 0.0

    print("Feature Engineering Complete.")
    # Drop raw strings before returning feature matrix
    cols_to_drop = ['name_s1', 'raw_name_s1', 'addr_s1', 'country_s1', 'pin_s1', 
                    'name_cand', 'raw_name_cand', 'addr_cand', 'country_cand', 'pin_cand',
                    'candidate_entity_ids'] # drop grouped ids if present
    for c in cols_to_drop:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
            
    return df
