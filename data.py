import pandas as pd
from datasets import load_dataset

# 1. Load primary 4-class data
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "datasets", "mental_heath_unbanlanced.csv")
test_path  = os.path.join(BASE_DIR, "datasets", "mental_health_combined_test.csv")

train_df = pd.read_csv(train_path)     # 49,612 rows
test_df  = pd.read_csv(test_path)      # 992 rows

# 2. Load GoEmotions
go = load_dataset("google-research-datasets/go_emotions", "simplified")["train"]

# 3. Extract happy/sadness/normal from GoEmotions
HAPPY = {"joy","amusement","gratitude","optimism","pride","relief","excitement"}
SAD   = {"sadness","grief","disappointment","remorse","embarrassment"}

def go_to_label(example):
    labels = [go.features["labels"].feature.int2str(i) for i in example["labels"]]
    if any(l in HAPPY for l in labels): return "happy"
    if any(l in SAD   for l in labels): return "sadness"
    if "neutral" in labels: return "normal"
    return None

go_rows = []
for ex in go:
    lbl = go_to_label(ex)
    if lbl:
        go_rows.append({"text": ex["text"], "status": lbl})

go_df = pd.DataFrame(go_rows)

# 4. Combine and rebalance
train_df['status'] = train_df['status'].astype(str).str.lower().str.strip()
go_df['status']    = go_df['status'].astype(str).str.lower().str.strip()

combined = pd.concat([train_df[['text', 'status']], go_df[['text', 'status']]], ignore_index=True)
# Target: ~8,000 per class. Oversample minority, undersample majority.
TARGET = 8000
balanced = combined.groupby("status").sample(n=TARGET, replace=True, random_state=42)
print("Balanced status counts:\n", balanced["status"].value_counts())