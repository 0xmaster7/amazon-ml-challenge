"""
smoke_tests.py — synthetic end-to-end checks before you trust a submission.

1. F-0.5 scorer edge cases (exact competition spec).
2. Synthetic FRENCH record test: France exists only in test, with zero
   training signal - verify French rows clean sanely, are NOT flagged as
   mojibake, and get blocked by the lexical layers.

Run: python smoke_tests.py   (no dataset or GPU needed)
"""
import pandas as pd
from data_cleaner import process_dataframe
from feature_engineering import has_mojibake
from blocker import LayeredBlocker
from scorer import f_05_per_entity


def test_scorer_edge_cases():
    # empty GT + empty pred -> 1.0
    assert f_05_per_entity({'S1-1': set()}, {'S1-1': set()}) == 1.0
    # empty GT + any pred -> 0.0
    assert f_05_per_entity({'S1-1': set()}, {'S1-1': {'S2-1'}}) == 0.0
    # non-empty GT + empty pred -> 0.0, no division error, no silent skip
    assert f_05_per_entity({'S1-1': {'S2-1'}}, {'S1-1': set()}) == 0.0
    # macro, not micro: perfect entity + failed entity average to 0.5
    s = f_05_per_entity({'S1-1': {'S2-1'}, 'S1-2': {'S2-2'}},
                        {'S1-1': {'S2-1'}, 'S1-2': set()})
    assert abs(s - 0.5) < 1e-9, s
    # spec example: pred 3, true 2, tp 2 -> 0.714
    s = f_05_per_entity({'S1-1': {'S2-47', 'S3-812'}},
                        {'S1-1': {'S2-47', 'S2-193', 'S3-812'}})
    assert abs(s - 0.714) < 0.001, s
    print("[PASS] F-0.5 scorer edge cases (incl. the spec's own worked example)")


def test_french_record():
    s1 = pd.DataFrame({
        'entity_id': ['S1-900001'],
        'business_name': ['Boulangerie Dupont SARL'],
        'business_address': ['12 Rue de la République, 69001 Lyon'],
        'country': ['France'],
    })
    s2 = pd.DataFrame({
        'entity_id': ['S2-900001'],
        'business_name': ['Boulangerie Dupont Société à responsabilité limitée'],
        'business_address': ['12 rue de la republique 69001 Lyon'],
        'country': ['FR'],
    })
    s3 = pd.DataFrame({
        'entity_id': ['S3-900001'],
        'business_name': ['Café Lumière'],
        'business_address': ['8 avenue des Champs-Élysées, 75008 Paris'],
        'country': ['France'],
    })
    s1, s2, s3 = process_dataframe(s1), process_dataframe(s2), process_dataframe(s3)

    # French text must survive cleaning enough to match
    assert 'boulangerie dupont' in s2.iloc[0]['clean_name'], s2.iloc[0]['clean_name']
    # French accents must survive to the embedder text
    assert 'République' in s1.iloc[0]['embed_text'], s1.iloc[0]['embed_text']
    # French names must NOT be flagged as mojibake garbage
    assert has_mojibake(s2.iloc[0]['business_name']) == 0
    # Country label drift FR vs France must normalize equal
    assert s1.iloc[0]['country_norm'] == s2.iloc[0]['country_norm'] == 'france'

    b = LayeredBlocker()
    b.layer1_exact_key_blocking(s1, s2, s3)
    b.layer2_tfidf_blocking(s1, s2, s3, sim_threshold=0.10)
    b.layer4_address_only(s1, s2, s3)
    b.layer4b_near_exact_address(s1, s2, s3)
    b.layer5_phone_key_blocking(s1, s2, s3)
    pairs = set(b.candidate_pairs.keys())
    assert ('S1-900001', 'S2-900001') in pairs, f"French true pair not blocked: {pairs}"
    print("[PASS] Synthetic French record: cleaned, normalized, not flagged as "
          "mojibake, and retrieved by blocking")


if __name__ == "__main__":
    test_scorer_edge_cases()
    test_french_record()
    print("\nAll smoke tests passed.")
