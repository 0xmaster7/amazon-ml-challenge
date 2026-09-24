import pandas as pd
import re

def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    
    # Remove all non-alphanumeric characters except spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def standardize_business_name(name):
    if not name:
        return ""
    
    # Common business suffixes to normalize (using word boundaries \b)
    replacements = {
        r'\bcorporation\b': 'corp',
        r'\blimited\b': 'ltd',
        r'\bprivate\b': 'pvt',
        r'\bcompany\b': 'co',
        r'\bincorporated\b': 'inc',
        r'\bllc\b': 'llc' # already lowercase from clean_text
    }
    
    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name)
        
    return re.sub(r'\s+', ' ', name).strip()

def standardize_address(address):
    if not address:
        return ""
    
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
        
    # Optional: Extract pin codes (5 digits for US, 6 digits for India/France)
    # We can just leave them in the text for now, TF-IDF will pick them up
    
    return re.sub(r'\s+', ' ', address).strip()

def process_dataframe(df):
    """Applies all cleaning steps to the dataframe"""
    print("Cleaning names...")
    df['clean_name'] = df['business_name'].apply(clean_text).apply(standardize_business_name)
    
    print("Cleaning addresses...")
    df['clean_address'] = df['business_address'].apply(clean_text).apply(standardize_address)
    
    return df

if __name__ == "__main__":
    print("Loading first 10 rows of Source 1...")
    # Read only first 10 rows to inspect visually
    df_s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=10)
    
    df_s1_clean = process_dataframe(df_s1.copy())
    
    # Display Before and After
    for i, row in df_s1_clean.iterrows():
        print("-" * 50)
        print(f"ORIGINAL NAME: {row['business_name']}")
        print(f"CLEANED NAME : {row['clean_name']}")
        print(f"ORIGINAL ADDR: {row['business_address']}")
        print(f"CLEANED ADDR : {row['clean_address']}")
