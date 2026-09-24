import pandas as pd
import re

# ============================================================
# CITY ALIAS MAP (India + France)
# ============================================================
CITY_ALIASES = {
    'bengaluru': 'bangalore', 'bombay': 'mumbai', 'calcutta': 'kolkata',
    'madras': 'chennai', 'trivandrum': 'thiruvananthapuram', 'cochin': 'kochi',
    'pondicherry': 'puducherry', 'baroda': 'vadodara', 'poona': 'pune',
    'simla': 'shimla', 'ooty': 'udhagamandalam', 'vizag': 'visakhapatnam',
    'marseilles': 'marseille', 'lyons': 'lyon',
}

def clean_text(text):
    """Basic scrub: mojibake fix, lowercase, strip punctuation, normalize whitespace."""
    if pd.isna(text) or text is None or not isinstance(text, str):
        return ""
    # Attempt mojibake reversal (Latin-1 misread as UTF-8)
    try:
        text = text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_pincode(address):
    """Extracts 5 or 6 digit pin codes for exact blocking."""
    if not address:
        return ""
    match = re.search(r'\b\d{5,6}\b', address)
    return match.group(0) if match else ""

def normalize_city(text):
    """Replaces known city aliases with canonical names."""
    if not text:
        return text
    for alias, canonical in CITY_ALIASES.items():
        text = re.sub(r'\b' + alias + r'\b', canonical, text)
    return text

def standardize_business_name(name):
    """Standardizes legal suffixes (English, Hindi, French)."""
    if not name:
        return ""
    replacements = {
        # English
        r'\bcorporation\b': 'corp', r'\blimited\b': 'ltd', r'\bprivate\b': 'pvt',
        r'\bcompany\b': 'co', r'\bincorporated\b': 'inc', r'\bllc\b': 'llc',
        r'\bassociates\b': 'assoc', r'\benterprises\b': 'ent',
        r'\binternational\b': 'intl', r'\bmanufacturing\b': 'mfg',
        r'\bbrothers\b': 'bros', r'\bindustries\b': 'ind',
        # French
        r'\bsociete anonyme\b': 'sa', r'\bsociete a responsabilite limitee\b': 'sarl',
        r'\bsociete par actions simplifiee\b': 'sas',
        r'\bentreprise unipersonnelle a responsabilite limitee\b': 'eurl',
        r'\bsarl\b': 'sarl', r'\bsas\b': 'sas', r'\beurl\b': 'eurl',
    }
    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name)
    return re.sub(r'\s+', ' ', name).strip()

def expand_abbreviations(name):
    """Expands abbreviations back to full form for a second-pass Jaro-Winkler comparison."""
    if not name:
        return ""
    expansions = {
        r'\bpvt\b': 'private', r'\bcorp\b': 'corporation', r'\bltd\b': 'limited',
        r'\binc\b': 'incorporated', r'\bco\b': 'company', r'\bst\b': 'street',
        r'\bave\b': 'avenue', r'\brd\b': 'road', r'\bhwy\b': 'highway',
        r'\bste\b': 'suite', r'\bapt\b': 'apartment', r'\bdr\b': 'drive',
        r'\bblvd\b': 'boulevard', r'\bintl\b': 'international',
        r'\bmfg\b': 'manufacturing', r'\bbros\b': 'brothers',
        r'\bind\b': 'industries', r'\bent\b': 'enterprises',
        r'\bassoc\b': 'associates',
    }
    for pattern, replacement in expansions.items():
        name = re.sub(pattern, replacement, name)
    return re.sub(r'\s+', ' ', name).strip()

def standardize_address(address):
    """Standardizes street terms and removes noisy landmark prefixes."""
    if not address:
        return ""
    address = re.sub(r'\b(near|opp|opposite|behind|beside)\b.*', '', address)
    replacements = {
        r'\broad\b': 'rd', r'\bstreet\b': 'st', r'\bavenue\b': 'ave',
        r'\bhighway\b': 'hwy', r'\bsuite\b': 'ste', r'\bapartment\b': 'apt',
        r'\broom\b': 'rm', r'\bnorth\b': 'n', r'\bsouth\b': 's',
        r'\beast\b': 'e', r'\bwest\b': 'w', r'\bdrive\b': 'dr',
        r'\bboulevard\b': 'blvd', r'\bfloor\b': 'fl', r'\bbuilding\b': 'bldg',
        # French
        r'\brue\b': 'rue', r'\bavenue\b': 'ave', r'\bboulevard\b': 'blvd',
        r'\bplace\b': 'pl', r'\bchemin\b': 'ch',
    }
    for pattern, replacement in replacements.items():
        address = re.sub(pattern, replacement, address)
    return re.sub(r'\s+', ' ', address).strip()

def extract_house_number(address):
    """Extracts leading numeric token (house/building number)."""
    if not address:
        return ""
    match = re.match(r'^(\d+)', str(address).strip())
    return match.group(1) if match else ""

def extract_street_tokens(address):
    """Extracts street-name tokens, ignoring house number and city-level words."""
    if not address:
        return ""
    # Remove leading house number
    addr = re.sub(r'^\d+\s*', '', str(address).strip())
    # Remove pincode
    addr = re.sub(r'\b\d{5,6}\b', '', addr)
    return addr.strip()

def process_dataframe(df):
    """Applies the full cleaning pipeline to a dataframe."""
    print("Cleaning basic text...")
    df['clean_name'] = df['business_name'].apply(clean_text).apply(standardize_business_name)
    df['clean_address'] = df['business_address'].apply(clean_text).apply(normalize_city).apply(standardize_address)

    print("Extracting features for Blocking...")
    df['extracted_pin'] = df['business_address'].apply(clean_text).apply(extract_pincode)
    df['name_first_token'] = df['clean_name'].apply(lambda x: x.split()[0] if x else "")
    df['house_number'] = df['clean_address'].apply(extract_house_number)
    df['street_tokens'] = df['clean_address'].apply(extract_street_tokens)
    
    # Expanded-abbreviation version of the name for second-pass JW
    df['expanded_name'] = df['clean_name'].apply(expand_abbreviations)

    return df
