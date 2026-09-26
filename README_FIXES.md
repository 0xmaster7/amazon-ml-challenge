# Fix pack - what to replace and why

Replace these files in the repo (paths mirror the repo layout):

## code/business_entity_resolution/src/data_cleaner.py
- clean_text() is now explicitly lexical-only; new make_embed_text() preserves
  raw scripts/accents for the multilingual embedder. FIX: Devanagari/French
  text was being erased to spaces before embedding.
- standardize_address(): landmark regex now removes only the landmark phrase
  (landmark word + up to 3 following words), not the whole rest of the address.
  FIX: "near sbi atm 12 mg rd bangalore" used to become "".
- extract_pincode(): takes the LAST 5-6 digit number (pincodes end addresses;
  leading numbers are house numbers). FIX: house numbers were used as PINs.
- New normalize_country() + country_norm column. FIX: "US" vs "USA" label
  drift silently broke the country_match feature.
- Adds raw_address column (landmark detection needs pre-cleaning text) and
  embed_text column.

## code/business_entity_resolution/src/blocker.py
- Layer 2 replaced: the commented-out pure-Python MinHash (too slow to ever
  run) is gone; new layer2_tfidf_blocking() is a vectorized sparse TF-IDF
  char-n-gram blocker. Same typo-safety-net job, actually runs.
- Layer 3 embeds embed_text (raw) instead of clean text. FIX: the multilingual
  model was being fed ASCII-stripped text.
- Memmap embedding cache now validated by a content-derived key (sidecar .key
  file), not byte size. FIX: same-row-count different data silently reused
  wrong embeddings.
- FAISS top_k 20 -> 30, threshold 0.55 -> 0.50 (recall is free at this stage).
- Embedding dimension now read from the model, not hardcoded 384.

## code/business_entity_resolution/src/feature_engineering.py
- Cross-encoder swapped to multilingual mmarco-mMiniLMv2-L12-H384-v1 (was
  English-only ms-marco-MiniLM), logits sigmoided to (0,1).
- Cross-encoder failure now RAISES with instructions; --no_cross_encoder flag
  disables it consistently. FIX: silent 0.0 fill caused train/inference
  distribution shift.
- has_landmark computed on raw_address. FIX: it was always 0 (cleaning had
  already removed the landmark words).
- country_match uses normalized labels.
- extract_domain_alpha wired into a real domain_name_sim feature (was dead
  code; website-as-name had no bridge).

## code/business_entity_resolution/src/train.py
- Threshold tuned on out-of-fold probabilities with the macro F-0.5 scorer
  (single global operating point), sweep 0.30-0.97 step 0.01. FIX: threshold
  from the single best fold was applied to the 5-model ensemble average;
  sweep used to cap at 0.90.
- NEW: blocking-recall report - overall, per layer, per entity. The strategy
  doc's "recall ceiling" is now measured, not assumed.
- LLM band brackets the tuned threshold (+/-0.15) instead of hardcoded
  0.35-0.65, and the LLM's macro F-0.5 delta is measured on validation.
- GT IDs stripped of whitespace before labeling. FIX: "S2-x, S2-y" GT could
  silently mislabel true pairs as negatives.
- New --no_cross_encoder flag.

## code/business_entity_resolution/src/predict.py
- LLM band brackets the loaded tuned threshold; --no_cross_encoder flag;
  removed the unused candidate_set build; prints test countries so France
  handling is visible.

## code/business_entity_resolution/src/llm_sniper.py
- Fallback path now returns a positional list of ints (was a pandas Series -
  index-alignment hazard for the caller).

## code/business_entity_resolution/requirements.txt
- Pinned versions; added scikit-learn, scipy, torch (were imported but
  missing); faiss-cpu instead of faiss-gpu so it installs anywhere (GPU search
  path uses torch, faiss is only the CPU fallback).

## Documentation_template.md
- Filled with the real methodology (was empty; it is a required deliverable).
  Update the numbers after your next training run.

## colab_runner.ipynb / kaggle_runner.ipynb
- Run commands updated to flags that actually exist (--data_dir/--output_dir;
  the old --checkpoint_dir/--epochs crashed instantly).

## NOT auto-fixed (deliberate)
- scale_pos_weight vs precision-heavy metric: kept, but worth an ablation run.
- Layer 4 still exact-equality (near-exact variant is future work; TF-IDF +
  FAISS cover most of it now).
- No fine-tuning code for the embedder (the hook existed but nothing trained
  it; that is a new feature, not a fix).
- test_cleaning.py left as a stale duplicate; delete it or ignore it.
- Cross-encoder choice: mmarco picked for size; BAAI/bge-reranker-v2-m3 is the
  stronger but heavier alternative if you have the GPU headroom.

## Kaggle readiness (added after your "running on Kaggle, too slow" message)
- train.py and predict.py now checkpoint every stage (cleaned data, candidate
  pairs, features) into output_dir/checkpoints/. Re-run any command with
  --resume and finished stages are skipped - a session cutoff costs you
  nothing. If you CHANGE cleaning/blocking/feature code, delete
  /kaggle/working/output/checkpoints first (stale checkpoints are not
  auto-invalidated).
- Embedding memmaps were already cached on disk; combined with the new
  content-key validation, a re-run after a cutoff skips the multi-hour encode
  safely (the old byte-size check could have silently reused WRONG embeddings).
- kaggle_runner.ipynb is now ONE working notebook: config cell at top
  (auto-detects student_resource/dataset/train vs dataset/train layout),
  install, train --resume, predict --resume, zip outputs. Delete
  colab_runner.ipynb from the repo - it called flags that never existed.
- GPU/CPU: requirements use faiss-cpu (installs everywhere); the GPU fast path
  uses torch matmul and is auto-detected, FAISS CPU is only the fallback.
- Speed reality check: the slow stages are (1) embedding encode, (2) TF-IDF,
  (3) cross-encoder over all candidate pairs, (4) row-wise feature building.
  All four are cached/checkpointed now, so the "way too long" pain hits once,
  not every run. For fast iteration use --resume --skip_llm; drop --skip_llm
  only for the final scoring run.
- LLM arbitration stays strictly on the borderline band (tuned threshold
  +/-0.15), and train.py now prints its validation F-0.5 delta - if that delta
  is negative, keep --skip_llm for predict too.

## Smoke-tested
The cleaner, TF-IDF blocking layer, Layer 1/4, and the full feature builder
were executed on synthetic US/India/France rows (landmark phrase removal keeps
house number + pincode, Devanagari survives embed_text, .in domains detected).
Full train/predict need your real dataset + GPU box, which I don't have -
watch the new BLOCKING RECALL printout on your first re-run; that's the
scoreboard for whether the fixes landed.

---

# v2 - strategy doc gap analysis (every item, mapped)

New files added in this version:
- src/scorer.py (metric extracted from train.py so light scripts don't need xgboost)
- src/error_analysis.py
- src/smoke_tests.py
- src/finetune_embedder.py

| Strategy item | Status |
|---|---|
| Macro per-entity F_0.5 with empty-GT edge cases | Already in fixpack (scorer); edge cases now asserted in smoke_tests.py incl. the spec's 0.714 example |
| Checkpointing / resume on Kaggle | Already in fixpack (train/predict --resume, memmap caches, notebook) |
| GPU/CPU autodetect, faiss-cpu | Already in fixpack |
| Raw text to embedder (mojibake fix) | Already in fixpack (embed_text) |
| Landmark regex keeps the rest of the address | Already in fixpack |
| PIN extraction = last 5-6 digit number | Already in fixpack |
| Country normalization (US/USA, FR/France) | Already in fixpack (country_norm) |
| Multilingual cross-encoder, no silent 0.0 fill | Already in fixpack |
| Layer 2 typo safety net that actually runs | Already in fixpack (TF-IDF char n-grams) |
| Embedding cache validity by content key | Already in fixpack |
| LLM sniper on borderline band only | Already in fixpack; band now tunable via --band_lo/--band_hi, default from tuned threshold +/- 0.15 |
| Self-validation of outputs | Already in fixpack (validate_outputs in predict.py) |
| StratifiedGroupKFold by entity country | NEWLY IMPLEMENTED: train.py folds now stratify by country (each fold sees France-like label drift mixes) |
| XGBoost seed ensembling | NEWLY IMPLEMENTED: --seeds 42 1337 2024 trains each fold across seeds and averages probabilities |
| scale_pos_weight ablation | NEWLY IMPLEMENTED: default ON; --no_spw runs the no-SPW variant for comparison |
| Per-country (stratified) thresholds | NEWLY IMPLEMENTED: --stratify tunes thresholds per country on OOF, stored as a dict in the pkl; predict.py applies them per row |
| GT assumption check: one pool ID claimed by two S1s | NEWLY IMPLEMENTED: train.py prints the GT conflict count; predict.py post-resolves conflicts (keeps highest-proba claim); --no_conflict_resolution if the check warns |
| Layer 4b near-exact address variant (token Jaccard >= 0.85 in pincode blocks) | NEWLY IMPLEMENTED in blocker.py |
| Layer 5 phone-key blocking | NEWLY IMPLEMENTED: 7+ digit phone runs extracted in data_cleaner.py, blocked in blocker.py, scored as phone_key_match feature |
| City/state tags (Indian cities + states, US state codes) | NEWLY IMPLEMENTED: city_tag/state_tag columns, city_match/state_match features (1/-1/0 semantics) |
| Dual-embedder recall (--embedder2) | NEWLY IMPLEMENTED in train.py + predict.py (cache-safe per model slug) |
| Error analysis script | NEWLY IMPLEMENTED: error_analysis.py buckets false merges (incl. chain trap + singleton hits), FNs by noise type, recall-ceiling misses, suggests LLM band |
| Synthetic French-record test | NEWLY IMPLEMENTED: smoke_tests.py checks French cleaning, accent survival to embedder, no false mojibake flag, FR/France normalization, blocking retrieval |
| Embedder fine-tuning (MultipleNegativesRankingLoss on GT pairs) | NEWLY IMPLEMENTED: finetune_embedder.py saves to output/finetuned_embedder, which Layer 3 auto-detects |
| Official validate_submission.py before every submit | CHECKLIST step: notebook section 7 runs it + smoke tests every time |
| Watch blocking-recall printout on first real run | CHECKLIST step: it's the scoreboard for whether fixes landed |
| LLM budget cap (< 5000 borderline pairs) | Already in fixpack (predict.py skips LLM above 5000) |

Not implemented, with reason:
- Nothing substantive. Every strategy item is either in code above or a process/checklist step that cannot be code.

---

# v2.1 hotfix (checkpoint NameError)

- train.py: save_ckpt/load_ckpt restored (the scorer refactor had moved them
  out) and macro_score_from_proba added to the scorer import. FIX: crashed
  with NameError at the LOADING DATA stage with --resume, and would have
  crashed during CV scoring even without it.
- scorer.py: now imports pandas (a moved helper needed it).
- Whole pack re-scanned for used-but-never-defined names (pyflakes over all
  10 source files) - no other instances. predict.py was never affected (it
  has its own checkpoint helpers).

---

# v2.2 hotfix (Layer 2 OOM kill on Kaggle)

- blocker.py layer2_tfidf_blocking rewritten memory-bounded: HashingVectorizer
  (no multi-GB vocabulary dict), two sliced passes (idf counts, then
  per-slice matching) so the full 2.2M-row pool matrix is never materialized,
  and per-entity hit accumulation pruned to ~top_k as it goes. Verified:
  identical retrieval behavior to the old implementation on 20k-row synthetic
  data at 0.25GB peak, 500/500 typo recall on distinct-name records.
- Per-layer blocking checkpointing: blocker progress (pairs + completed
  layers) is saved after every layer. A kill during Layer 2 no longer loses
  Layer 1's pairs - re-run with --resume and completed layers are skipped.
  (Layer 3 always re-runs so the feature stage gets embeddings attached, but
  its memmap cache skips re-encoding.)
- Notebook + one-cell runner: official validator args corrected to
  --matching/--candidate/--test-dir.

---

# v2.3 hotfix (next OOM: feature stage + leaner defaults)

- feature_engineering.py: features now build in 200k-pair chunks - the
  string-heavy merged frame no longer covers the whole candidate set at once;
  cross-encoder model loads once and is reused across chunks. predict.py gets
  the same bounding automatically (same function).
- blocker.py Layer 2 defaults lowered (pool_slice 100k, s1_batch 10k);
  Layer 3 GPU search slices lowered (pool 400k, s1 batch 1000) so the fp16
  similarity matrix stays ~1GB on the T4.
- train.py: feature matrix cast to float32 for XGBoost; blocking-stage
  frames and embedder memmaps freed before training.
- Rerun: NO wipe needed - blocker per-layer checkpoints from the last run
  resume cleanly. Just re-pull the branch and re-run the cell.

---

# v2.4 hotfix (Layer 2 died on its FIRST pool slice)

- Root cause: the 2^21 hash space is DENSER than the true n-gram vocabulary,
  so hash collisions inflated the sparse product's nonzero count before
  thresholding could filter - that intermediate OOM'd the kernel.
- Hash space raised to 2^24 (collision noise drops ~8x); sub-threshold
  entries are now zeroed + compacted on the CSR BEFORE any COO
  materialization; pool deduped by entity_id.
- Verified: 500/500 typo recall on realistic distinct-name synthetic data,
  0.45GB peak. Pool dedup by entity_id is the only safe shrink - two rows
  with identical text but different entity_ids are still distinct candidates.

---

# v2.5 hotfix (Layer 2 OOM, third strike - structural fix)

Root cause was structural, not tuning. Three full copies of the data were
alive at the matching step: the original cleaned frames (df_s2/df_s3, ~19
object columns each), a full-frame pd.concat copy of them, and pool_texts/
s1_texts lists copying all the text again - plus the entire S1 TF-IDF matrix
Q resident in RAM. On a 10.4M-row pool that exceeds 13GB before any slice
math runs.

Fixes in blocker.py:
- Every layer now concatenates ONLY the columns it uses (3 max), not whole
  frames: layers 1, 3, 4, 4b, 5 and 2.
- Layer 2 builds text slices on demand - no pool_texts/s1_texts lists.
- The S1 matrix Q is written to disk in .npz chunks and loaded one chunk at
  a time during matching - constant RAM regardless of source1 size.
- (from v2.4) 2^24 hash buckets + sub-threshold compaction before COO.
Verified: 500/500 typo recall, multi-chunk path exercised, scratch files
cleaned up, 0.44GB sandbox peak. Layer 5 phone blocking confirmed working.
