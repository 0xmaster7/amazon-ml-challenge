# Methodology: Business Entity Resolution

## 1. Methodology used

Two-stage entity resolution pipeline:

1. **Blocking (candidate generation):** a layered, unioned candidate generator
   produces `candidate_pairs.tsv`. Recall is optimized here, not precision -
   `candidate_pairs.tsv` is not scored, so over-generating candidates is cheap,
   while every true match missed at this stage is a permanent cap on the final
   score.
2. **Matching (final inference):** an XGBoost classifier over hand-built string,
   address, structural, and neural-similarity features scores each candidate
   pair, and an out-of-fold-calibrated decision threshold converts scores into
   the final `matching_results.tsv`. A small open-weights LLM arbitrates only
   pairs the classifier is unsure about.

The evaluation metric is macro-averaged F_0.5 (precision weighted 2x over
recall), computed per Source 1 entity including singleton credit. Every design
decision below traces back to that metric: blocking maximizes recall, matching
is deliberately conservative, and the decision threshold is tuned directly on
the competition formula rather than accuracy.

## 2. Candidate generation / blocking strategy

Four layers, unioned (each layer catches what the others miss):

- **Layer 1 - exact key blocking:** normalized name first token + extracted
  pincode (last 5-6 digit number in the address; pincodes sit at the end).
  Near-zero false negatives for clean duplicates.
- **Layer 2 - TF-IDF character n-gram blocking:** sparse TF-IDF over char
  3-5-grams of the normalized name+address, cosine similarity via vectorized
  sparse matrix products in pool slices. This is the typo/OCR-corruption
  safety net (`Wilblims` vs `Williams`).
- **Layer 3 - multilingual embedding retrieval:**
  `paraphrase-multilingual-MiniLM-L12-v2` embeddings of the RAW record text
  (scripts and accents preserved - the multilingual model must actually see
  Devanagari and French text), top-k=30 nearest neighbors above cosine 0.50
  via GPU matmul with FAISS CPU fallback. Multilingual because the test set
  adds France, unseen in training.
- **Layer 4 - address-only fallback:** exact match on standardized addresses,
  ignoring the name, for rows where the name is garbage but the address is
  clean.

Blocking recall (found GT pairs / total GT pairs, overall and per layer) is
reported during training; it is the upper bound on final recall and the main
diagnostic for where matches are lost.

## 3. Model architecture and feature engineering

**Primary classifier:** XGBoost (histogram tree method, GPU), 5-fold
GroupKFold grouped by Source 1 entity so an entity never straddles train and
validation. The decision threshold is tuned on out-of-fold probabilities
against the exact macro F_0.5 competition formula, sweeping 0.30-0.97.

**Features (~28):**

- Name similarity: raw and suffix-stripped exact match, Jaro-Winkler,
  token-sort and token-set ratios, character-trigram cosine,
  abbreviation-expanded Jaro-Winkler, name length difference.
- Neural similarity: bi-encoder cosine (reused from blocking) and a
  MULTILINGUAL cross-encoder score (`mmarco-mMiniLMv2-L12-H384-v1`, sigmoided
  logits) fed in as an ordinary feature - semantic understanding inside a
  debuggable GBM.
- Address: full-address fuzzy ratio, house-number match, pincode match,
  street-token Jaccard, numeric-token ratio, address-missing flag,
  landmark-reference flags (computed on the raw address, before cleaning
  removes landmark phrases).
- Website-as-name: URL flags plus a domain-token-to-name similarity bridge.
- Structural: normalized country label equality (never one-hot - France is
  unseen in training), source-pair type, mojibake flags, blocking-layer
  provenance (which layers caught the pair, FAISS rank/score).

**LLM arbitration ("sniper"):** Qwen2.5-7B-Instruct (4-bit, Apache 2.0 -
compliant with the MIT/Apache-2.0 license rule; Llama 3 is not) arbitrates
only pairs within +/-0.15 of the tuned threshold. Its macro F_0.5 delta is
measured on validation before it is allowed to affect test predictions; if the
delta is negative, inference runs with `--skip_llm`.

## 4. Other relevant information

- **Data cleaning:** mojibake reversal, legal-suffix standardization
  (English + French forms), address street-term standardization, city aliases,
  landmark-phrase removal (phrase only, not the rest of the address).
- **Output compliance:** `predict.py` self-validates every submission rule
  before writing: one row per Source 1 entity (singletons included with empty
  lists), no duplicate IDs within a list, only existing S2/S3 IDs, and
  `matching_results.tsv` is a strict subset of `candidate_pairs.tsv`.
- **No external data** is used anywhere; only the provided TSVs and pretrained
  open-weights models within the size/license constraints.
- **Reproducibility:** dependencies are pinned in requirements.txt; both output
  files regenerate from the raw data with `train.py` then `predict.py`.

## 5. Known limitations / future work

- Fine-tuning the bi-encoder on ground-truth pairs (hook exists in the code)
  should lift Layer 3 recall further.
- Layer 4 is exact-equality only; a token-Jaccard variant would catch
  single-token address typos.
- `scale_pos_weight` vs the precision-heavy metric deserves a proper ablation.
