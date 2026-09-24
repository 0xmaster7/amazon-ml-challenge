import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    try:
        text = text.encode("latin-1").decode("utf-8")
    except:
        pass

    """Basic scrub: lowercase, strip punctuation except spaces, normalize whitespace"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_pincode(address):
    """Extracts 5 or 6 digit pin codes for exact blocking"""
    if not address:
        return ""
    # Look for 6-digit (India/France) or 5-digit (US) numbers
    match = re.search(r'\b\d{5,6}\b', address)
    return match.group(0) if match else ""

def standardize_business_name(name):
    """Standardizes legal suffixes"""
    if not name:
        return ""
    
    replacements = {
        r'\bcorporation\b': 'corp',
        r'\blimited\b': 'ltd',
        r'\bprivate\b': 'pvt',
        r'\bcompany\b': 'co',
        r'\bincorporated\b': 'inc',
        r'\bllc\b': 'llc'
    }
    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name)
    return re.sub(r'\s+', ' ', name).strip()

def standardize_address(address):
    """Standardizes street terms and removes noisy landmark prefixes"""
    if not address:
        return ""
    
    # Strip common Indian landmark noise
    address = re.sub(r'\b(near|opp|opposite|behind|beside)\b.*', '', address)
    
    replacements = {
        r'\broad\b': 'rd',
        r'\bstreet\b': 'st',
        r'\bavenue\b': 'ave',
        r'\bhighway\b': 'hwy',
        r'\bsuite\b': 'ste',
        r'\bapartment\b': 'apt',
        r'\broom\b': 'rm',
        r'\bnorth\b': 'n',
        r'\bsouth\b': 's',
        r'\beast\b': 'e',
        r'\bwest\b': 'w',
    }
    for pattern, replacement in replacements.items():
        address = re.sub(pattern, replacement, address)
        
    return re.sub(r'\s+', ' ', address).strip()

def process_dataframe(df):
    """Applies the full cleaning pipeline to a dataframe"""
    print("Cleaning basic text...")
    df['clean_name'] = df['business_name'].apply(clean_text).apply(standardize_business_name)
    df['clean_address'] = df['business_address'].apply(clean_text).apply(standardize_address)
    
    print("Extracting features for Blocking...")
    df['extracted_pin'] = df['business_address'].apply(clean_text).apply(extract_pincode)
    
    # First token of the name (useful for exact blocking)
    df['name_first_token'] = df['clean_name'].apply(lambda x: x.split()[0] if x else "")
    
    return df
