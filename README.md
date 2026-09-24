# 🎯 Amazon ML Challenge: Business Entity Resolution

Based on the problem statement, here is a complete, easy-to-understand breakdown of exactly what we need to build, the constraints we have to follow, and the strategy we should use.

## 1. The Core Problem
You are given three lists of businesses. 
* **Source 1** is your "Master List" (deduplicated).
* **Source 2** and **Source 3** are messy, noisy lists containing typos, abbreviations, and missing data.

**Your Goal:** For every single business in Source 1, you must search through Source 2 and Source 3 and find all the records that actually refer to that exact same real-world business. 
* *Note: A Source 1 business might have zero matches (a "singleton"), one match, or many matches.*

## 2. The Required Two-Step Pipeline
The problem statement explicitly requires us to generate two output files, which implies we must build a standard two-step Entity Resolution pipeline:

### Step 1: Blocking (Candidate Generation)
* **The Problem:** If Source 1 has 100,000 rows and Source 2 has 100,000 rows, comparing every single row to each other would require 10 Billion comparisons. This would take weeks to compute.
* **The Solution:** We build a fast, rough algorithm (like TF-IDF, BM25 search, or basic vector embeddings). For every Source 1 business, it quickly retrieves the top 10 to 50 "most likely" matches from Sources 2 and 3. 
* **Output:** This generates `candidate_pairs.tsv` (required by the judges to see how good our initial filtering was).

### Step 2: Final Matching (Inference)
* **The Problem:** The fast blocking step will pull in a lot of false positives (businesses that sound similar but aren't the same).
* **The Solution:** We take only the candidates found in Step 1 and pass them through a highly accurate, heavy ML model (like XGBoost, a Cross-Encoder Neural Network, or a small LLM). This model looks deeply at the Name, Address, and Country and makes a final `Yes/No` decision on whether they are truly a match.
* **Output:** This generates `matching_results.tsv` (this is the file that actually gets scored on the leaderboard).

## 3. How We Are Scored (F-0.5 Score)
This is a massive hint for our strategy. The competition uses the **F-0.5 Score**, which weights **Precision 2x more than Recall**.
* **What this means:** The judges penalize you *heavily* for False Positives (guessing two businesses are the same when they aren't). 
* **Our Strategy:** Our final matching model needs to be **conservative**. If the model is only 50% sure two businesses match, it is better to guess that they DO NOT match. Correctly predicting that a business has zero matches (a "singleton") gives us a perfect 1.0 score for that row.

## 4. The Data & The "Trap"
* **The Data:** Tab-separated (`.tsv`). Contains `entity_id`, `business_name`, `business_address`, and `country`. 
* **The Noise:** The addresses and names are terribly formatted. We will see "Corp" vs "Corporation", "Rd" vs "Road", landmarks instead of street names ("Near SBI ATM"), and missing pin codes.
* **The Trap:** The training data only contains businesses in the **US** and **India**. However, the hidden Test data will introduce a brand new country: **France**. Our model cannot rely *only* on US/India address formats; it must be smart enough to generalize to French addresses without having seen them in training!

## 5. Strict Constraints (Must Follow!)
1. **NO EXTERNAL DATA:** This is strictly prohibited. We **cannot** use Google Maps APIs, geocoding libraries, or external government databases to clean up the addresses. We can only use the provided text. (Using them results in instant disqualification).
2. **Model Size Limit (8 Billion Parameters):** We cannot use massive state-of-the-art LLMs (like GPT-4, Claude, or Llama 3 70B). The final model must be open-source (MIT/Apache 2.0) and under 8B parameters. 
   * *What we can use:* Traditional ML (XGBoost/LightGBM with string similarity features), BERT-style Encoders, or small LLMs like Llama-3-8B or Qwen-7B.
3. **Exact Output Formatting:** The `.tsv` files must be formatted exactly as requested, or the submission will fail. (They provided a `validate_submission.py` script we will use to test our files before uploading).

## 6. The Final Deliverable
When the competition ends, we must submit a specific `.zip` file containing:
* Our two output `.tsv` files.
* Our full, runnable source code (which we are already scaffolding!).
* A `requirements.txt` file.
* A written Methodology document explaining our Blocking and Matching strategies.
