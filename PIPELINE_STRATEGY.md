# Amazon ML Challenge: Execution Strategy & Data Cleaning

## 1. The Multi-Layered Blocking Strategy (Stage 1)
To handle extreme noise (DBAs, missing addresses, typos, and multilingual France/India data), we use a **Layered Blocking Pipeline**, unioning four cheap passes into `candidate_pairs.tsv`.

### Layer 1: Exact / Near-Exact Key Blocking
* **How:** Normalize name + first token, combined with pincode/city if present.
* **Why:** Catches the "easy" matches instantly for free.

### Layer 2: MinHash / LSH (Locality Sensitive Hashing)
* **How:** Break strings into character n-grams and use LSH.
* **Why:** The safety net for heavy typos and OCR corruption (`Wilblims` vs `Williams`). 

### Layer 3: FAISS + Multilingual Embeddings
* **How:** Pass text through a fast multilingual sentence embedder (`paraphrase-multilingual-MiniLM-L12-v2`) and use FAISS.
* **Why:** Catches DBA (Doing Business As) swaps, website-as-name, and cross-lingual transliterations.

### Layer 4: Address-Only Fallback
* **How:** Strict block on exact/near-exact addresses while ignoring the business name.
* **Why:** Catches cases where the name is garbage but the address is clean.

---

## 2. The Final Matching Model (Stage 2)
Stage 2 acts as a ruthless filter to optimize the F-0.5 score (killing False Positives). We will use a **Stacked Ensemble Approach**.

### The Primary Classifier: XGBoost / LightGBM
* **Why:** It is fast, highly debuggable for the methodology doc, and we can explicitly calibrate the decision threshold to favor precision for the F-0.5 metric.
* **The Features:**
  * String distances: Jaro-Winkler, token-sort Levenshtein.
  * Address features: Exact pincode match boolean, street number match, fuzzy string match.
  * Meta flags: `name_is_url` boolean, `address_missing` boolean.
  * **The Secret Weapon (Cross-Encoder):** We run a small multilingual cross-encoder, but instead of using it on its own, we feed its output probability *into XGBoost as just another feature*. This gives XGBoost semantic understanding!

### The LLM Sniper (Borderline Arbitration)
* **Why:** LLMs are too slow for millions of pairs and bad at probability calibration. We will ONLY use an LLM for borderline cases where XGBoost is unsure (e.g., scores between 0.35 and 0.65).
* **⚠️ CRITICAL LICENSE TRAP:** We **CANNOT use Llama 3**. The rules require an MIT or Apache-2.0 license. Llama uses a custom Meta license and could cause instant disqualification. 
* **The Solution:** We will use **Qwen2.5-7B-Instruct** or **Mistral-7B-Instruct** (both strictly Apache 2.0).

---

## 3. Data Cleaning & Normalization Strategy
Because Layer 1 and 2 rely on lexical matching, data cleaning is the foundation.

### A. The Basic Scrub
* Lowercasing & punctuation stripping (`St. John's, Inc.` -> `st johns inc`). Replace `NaN` with `""`.

### B. Business Name Standardization
* Standardize legal suffixes (`corporation` → `corp`, `private limited` → `pvt ltd`).

### C. Address Standardization (Regex)
* `road` → `rd`, `street` → `st`, `avenue` → `ave`, `suite`/`apartment` → `ste`/`apt`.

### D. Country-Specific Noise (US, India, France)
* Strip contextual Indian landmark words ("Opposite", "Near").
* Extract 5-digit (US) and 6-digit (India/France) pins into an `extracted_pin` column.