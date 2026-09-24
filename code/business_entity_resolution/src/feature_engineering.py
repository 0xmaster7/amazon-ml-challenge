import pandas as pd
import re
from rapidfuzz import fuzz, distance

def is_website(text):
    if not text: return 0
    return 1 if re.search(r'(\.com|\.net|\.org|www\.)', str(text).lower()) else 0

def extract_numbers(text):
    if not text: return set()
    return set(re.findall(r'\d+', str(text)))

def get_numeric_match_ratio(str1, str2):
    nums1 = extract_numbers(str1)
    nums2 = extract_numbers(str2)
    if not nums1 and not nums2:
        return 1.0 # Both have no numbers
    if not nums1 or not nums2:
        return 0.0
    intersection = len(nums1.intersection(nums2))
    return intersection / max(len(nums1), len(nums2))

def build_features_for_pairs(df_pairs, df_s1, df_pool):
    """
    Takes a dataframe of candidate pairs (source1_id, candidate_id) 
    and merges the raw text from S1 and the Candidate pool to compute XGBoost features.
    """
    print("Merging text for feature engineering...")
    
    # Merge S1 text
    df = df_pairs.merge(df_s1[['entity_id', 'clean_name', 'clean_address', 'country']], 
                        left_on='source1_entity_id', right_on='entity_id', how='left')
    df.rename(columns={'clean_name': 'name_s1', 'clean_address': 'addr_s1', 'country': 'country_s1'}, inplace=True)
    df.drop('entity_id', axis=1, inplace=True)
    
    # Merge Candidate text
    df = df.merge(df_pool[['entity_id', 'clean_name', 'clean_address', 'country']], 
                  left_on='candidate_entity_id', right_on='entity_id', how='left')
    df.rename(columns={'clean_name': 'name_cand', 'clean_address': 'addr_cand', 'country': 'country_cand'}, inplace=True)
    df.drop('entity_id', axis=1, inplace=True)
    
    # Fill NAs to prevent math crashes
    df.fillna("", inplace=True)

    print("Computing String Similarities (RapidFuzz)...")
    
    # Name Similarity Features
    df['name_exact_match'] = (df['name_s1'] == df['name_cand']).astype(int)
    df['name_jaro_winkler'] = df.apply(lambda row: distance.JaroWinkler.normalized_similarity(row['name_s1'], row['name_cand']), axis=1)
    df['name_token_sort_ratio'] = df.apply(lambda row: fuzz.token_sort_ratio(row['name_s1'], row['name_cand']) / 100.0, axis=1)
    df['name_token_set_ratio'] = df.apply(lambda row: fuzz.token_set_ratio(row['name_s1'], row['name_cand']) / 100.0, axis=1)
    
    # Flags and Meta
    df['is_website_s1'] = df['name_s1'].apply(is_website)
    df['is_website_cand'] = df['name_cand'].apply(is_website)
    df['address_missing'] = ((df['addr_s1'] == "") | (df['addr_cand'] == "")).astype(int)
    df['country_match'] = (df['country_s1'] == df['country_cand']).astype(int)
    df['name_length_diff'] = abs(df['name_s1'].str.len() - df['name_cand'].str.len())
    
    # Address Similarity Features
    df['addr_fuzzy_ratio'] = df.apply(lambda row: fuzz.ratio(row['addr_s1'], row['addr_cand']) / 100.0 if row['address_missing'] == 0 else 0, axis=1)
    df['addr_numeric_match_ratio'] = df.apply(lambda row: get_numeric_match_ratio(row['addr_s1'], row['addr_cand']), axis=1)

    # TODO: Add Stage 1 Embedding Cosine Sim, Cross-Encoder Score, Block Method Origin
    
    print("Feature Engineering Complete.")
    return df
