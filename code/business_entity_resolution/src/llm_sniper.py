import pandas as pd
from transformers import pipeline

class LLMSniper:
    def __init__(self):
        print("Loading Qwen2.5-7B-Instruct for Arbitration...")
        try:
            # We load in 4-bit to fit on a Kaggle T4 GPU
            self.pipe = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-7B-Instruct",
                model_kwargs={"load_in_4bit": True},
                device_map="auto"
            )
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            self.pipe = None

    def arbitrate(self, df_ambiguous):
        """
        Takes a dataframe of borderline pairs (e.g., XGBoost probability between 0.35 and 0.65)
        and asks the LLM to make the final Yes/No call.
        """
        if self.pipe is None:
            print("LLM not loaded. Falling back to XGBoost scores.")
            return df_ambiguous['xgb_prob'] > 0.5

        results = []
        for _, row in df_ambiguous.iterrows():
            prompt = (
                "You are an expert at Entity Resolution. Determine if the following two business records refer to the exact same real-world entity.\n\n"
                f"Record 1: Name: '{row['name_s1']}', Address: '{row['addr_s1']}'\n"
                f"Record 2: Name: '{row['name_cand']}', Address: '{row['addr_cand']}'\n\n"
                "Are they the same business? Answer strictly with YES or NO."
            )
            
            messages = [
                {"role": "system", "content": "You are a highly precise entity resolution bot."},
                {"role": "user", "content": prompt}
            ]
            
            out = self.pipe(messages, max_new_tokens=5, temperature=0.0)
            response = out[0]['generated_text'][-1]['content'].strip().upper()
            
            if "YES" in response:
                results.append(1)
            else:
                results.append(0)
                
        return results
