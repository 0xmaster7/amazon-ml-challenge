import pandas as pd
import re
import os

def mem_rss():
    # Current process RSS in GB, no dependencies (reads /proc).
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) / 1e6
    except Exception:
        return -1.0
    return -1.1


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

# ============================================================
# COUNTRY NORMALIZATION
# country_match in feature engineering compares these labels,
# so label drift ("US" vs "USA") must not break equality.
# France never appears in training - keep the label, never one-hot.
# ============================================================
COUNTRY_MAP = {
    'us': 'us', 'usa': 'us', 'u s': 'us', 'u s a': 'us',
    'united states': 'us', 'united states of america': 'us', 'america': 'us',
    'india': 'india', 'in': 'india', 'bharat': 'india', 'republic of india': 'india',
    'france': 'france', 'fr': 'france', 'french republic': 'france',
}

def normalize_country(country):
    if not country or (isinstance(country, float) and pd.isna(country)):
        return ""
    key = re.sub(r'\s+', ' ', str(country).strip().lower().replace('.', ''))
    return COUNTRY_MAP.get(key, key)

def clean_text(text):
    """Basic scrub for LEXICAL layers: mojibake fix, lowercase, strip punctuation.
    Do NOT feed this to the multilingual embedder - it erases every non-[a-z0-9]
    character, which deletes Devanagari and French accents. Use embed_text for that."""
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

def make_embed_text(name, address):
    """Lightly-normalized RAW text for the multilingual embedder.
    Preserves non-Latin scripts and accents; only fixes mojibake and whitespace."""
    def fix(t):
        if pd.isna(t) or t is None:
            return ""
        t = str(t)
        try:
            t = t.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        return re.sub(r'\s+', ' ', t).strip()
    return (fix(name) + " " + fix(address)).strip()

def extract_pincode(address):
    """Extracts 5 or 6 digit pin codes. Takes the LAST 5-6 digit number:
    pincodes sit at the end of addresses, leading numbers are usually
    house/plot numbers."""
    if not address:
        return ""
    matches = re.findall(r'\b\d{5,6}\b', address)
    return matches[-1] if matches else ""

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


def remove_landmarks(text):
    """Removes landmark phrases from RAW address text (commas still intact).
    A landmark phrase runs from the landmark word to the next comma/semicolon,
    or to end of string if none. FIX: the old version ran AFTER clean_text had
    stripped commas, so it could not tell where the landmark phrase ended and
    ate the house number too ("near sbi atm 12 mg rd" lost the "12")."""
    if not text:
        return ""
    return re.sub(r'\b(?:near|opp|opposite|behind|beside)\b[^,;]*[,;]?', ' ', str(text), flags=re.IGNORECASE)

def standardize_address(address):
    """Standardizes street terms. Landmark phrases are removed earlier,
    on the raw text, by remove_landmarks()."""
    if not address:
        return ""
    replacements = {
        r'\broad\b': 'rd', r'\bstreet\b': 'st', r'\bavenue\b': 'ave',
        r'\bhighway\b': 'hwy', r'\bsuite\b': 'ste', r'\bapartment\b': 'apt',
        r'\broom\b': 'rm', r'\bnorth\b': 'n', r'\bsouth\b': 's',
        r'\beast\b': 'e', r'\bwest\b': 'w', r'\bdrive\b': 'dr',
        r'\bboulevard\b': 'blvd', r'\bfloor\b': 'fl', r'\bbuilding\b': 'bldg',
        # French
        r'\brue\b': 'rue', r'\bplace\b': 'pl', r'\bchemin\b': 'ch',
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
    """Extracts street tokens minus leading house number and pincode.
    NOTE: city tokens are NOT stripped here (city names appear inside the
    address string); downstream token-overlap features therefore include
    city similarity. Acceptable, but be aware when reading feature importances."""
    if not address:
        return ""
    addr = re.sub(r'^\d+\s*', '', str(address).strip())
    addr = re.sub(r'\b\d{5,6}\b', '', addr)
    return addr.strip()

# ============================================================
# CITY / STATE / PHONE-KEY DETECTION
# Used for city_match / state_match features and the phone/tax-ID
# blocking layer (strategy doc: Indian records sometimes tuck phone
# or registration numbers into the address string).
# ============================================================
KNOWN_CITIES = set(CITY_ALIASES.keys()) | set(CITY_ALIASES.values()) | {
    'delhi', 'new delhi', 'hyderabad', 'ahmedabad', 'jaipur', 'lucknow',
    'surat', 'kanpur', 'nagpur', 'indore', 'bhopal', 'patna', 'ludhiana',
    'agra', 'nashik', 'faridabad', 'meerut', 'rajkot', 'varanasi', 'noida',
    'gurugram', 'gurgaon', 'chandigarh', 'coimbatore', 'kochi', 'guwahati',
    'bhubaneswar', 'thiruvananthapuram', 'visakhapatnam', 'vijayawada',
    'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'dallas',
    'san francisco', 'seattle', 'boston', 'austin', 'denver', 'atlanta',
    'miami', 'portland', 'minneapolis', 'detroit', 'philadelphia',
    'paris', 'lyon', 'marseille', 'toulouse', 'nice', 'nantes', 'bordeaux',
    'lille', 'strasbourg', 'rennes', 'montpellier',
}

INDIAN_STATES = {
    'maharashtra', 'karnataka', 'tamil nadu', 'telangana', 'gujarat',
    'rajasthan', 'uttar pradesh', 'west bengal', 'kerala', 'punjab',
    'madhya pradesh', 'bihar', 'odisha', 'assam', 'haryana', 'delhi',
    'andhra pradesh', 'jharkhand', 'chhattisgarh', 'uttarakhand', 'goa',
}

US_STATE_CODES = {
    'al','ak','az','ar','ca','co','ct','de','fl','ga','hi','id','il','in','ia',
    'ks','ky','la','me','md','ma','mi','mn','ms','mo','mt','ne','nv','nh','nj',
    'nm','ny','nc','nd','oh','ok','or','pa','ri','sc','sd','tn','tx','ut','vt',
    'va','wa','wv','wi','wy','dc',
}

def detect_city(text):
    """Returns the canonical city name if one appears in the text, else ''."""
    if not text:
        return ""
    t = ' ' + str(text).lower() + ' '
    for city in KNOWN_CITIES:
        if (' ' + city + ' ') in t or re.search(r'\b' + re.escape(city) + r'\b', t):
            return CITY_ALIASES.get(city, city)
    return ""

def detect_state(text):
    """Returns a state tag: Indian state name or US state code, else ''."""
    if not text:
        return ""
    t = str(text).lower()
    for st in INDIAN_STATES:
        if re.search(r'\b' + re.escape(st) + r'\b', t):
            return st
    tokens = set(re.findall(r'\b[a-z]{2}\b', t))
    hit = tokens & US_STATE_CODES
    return sorted(hit)[0] if hit else ""

def extract_phone_keys(text):
    """Long digit runs (7+) from the RAW address/record: possible phone,
    tax-ID, or registration numbers embedded in the text."""
    if not text:
        return []
    return re.findall(r'\d{7,}', re.sub(r'[\s\-()+]', '', str(text)))

def process_dataframe(df):
    """Applies the full cleaning pipeline to a dataframe."""
    print("Cleaning basic text...")
    df['clean_name'] = df['business_name'].apply(clean_text).apply(standardize_business_name)
    df['clean_address'] = df['business_address'].apply(remove_landmarks).apply(clean_text).apply(normalize_city).apply(standardize_address)

    print("Extracting features for Blocking...")
    df['extracted_pin'] = df['business_address'].apply(clean_text).apply(extract_pincode)
    df['name_first_token'] = df['clean_name'].apply(lambda x: x.split()[0] if x else "")
    df['house_number'] = df['clean_address'].apply(extract_house_number)
    df['street_tokens'] = df['clean_address'].apply(extract_street_tokens)

    # Expanded-abbreviation version of the name for second-pass JW
    df['expanded_name'] = df['clean_name'].apply(expand_abbreviations)

    # Raw-preserved text for the multilingual embedder (scripts/accents intact)
    df['embed_text'] = [make_embed_text(n, a) for n, a in
                        zip(df['business_name'], df['business_address'])]

    # Raw lowercased address for landmark detection (cleaning removes landmarks,
    # so landmark flags must be computed BEFORE that)
    df['raw_address'] = df['business_address'].fillna("").astype(str).str.lower()

    # Normalized country label for equality features
    df['country_norm'] = df['country'].apply(normalize_country)

    # City / state tags for match features (strategy doc city+state features)
    df['city_tag'] = df['clean_address'].apply(detect_city)
    df['state_tag'] = df['raw_address'].apply(detect_state)

    # Phone / tax-ID keys from the RAW record text (name+address) for blocking.
    # Stored as a space-joined string, not a list - millions of tiny list
    # objects cost ~600MB of pure overhead on a 10M-row pool.
    df['phone_keys'] = [" ".join(extract_phone_keys(n) + extract_phone_keys(a))
                        for n, a in zip(df['business_name'], df['business_address'])]

    return slim_frame(df)


# Columns no stage needs after cleaning is done (raw_address keeps the
# pre-clean address; raw text for the embedder lives in embed_text).
_DROP_AFTER_CLEAN = ['business_address']
# Low-cardinality columns: category dtype replaces a Python string object
# per row with a small int code - the single biggest memory lever on
# 10M-row frames.
_CATEGORICAL_COLS = ['country', 'country_norm', 'city_tag', 'state_tag',
                     'extracted_pin', 'name_first_token', 'house_number']


def slim_frame(df):
    # Shrink a cleaned frame's RAM footprint. Idempotent - safe to re-apply
    # to frames loaded from an old checkpoint.
    for c in _DROP_AFTER_CLEAN:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    if 'phone_keys' in df.columns and df['phone_keys'].map(lambda v: isinstance(v, list)).any():
        df['phone_keys'] = df['phone_keys'].map(
            lambda v: " ".join(v) if isinstance(v, list) else (v if isinstance(v, str) else ""))
    for c in _CATEGORICAL_COLS:
        if c in df.columns and str(df[c].dtype) != 'category':
            df[c] = df[c].astype('category')
    return df
