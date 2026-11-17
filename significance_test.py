# ============================================================
# FULL PIPELINE (runs from scratch, nothing pre-saved needed)
# Fine-tunes XLM-RoBERTa-base AND AfroXLMR-base on a 70/30 split
# for seeds 42, 123, 2024 -> mean ± std + McNemar's test.
# Run in Google Colab with a GPU runtime (Runtime > Change runtime type).
# ============================================================

# ---- 0. Install dependencies -------------------------------
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "datasets", "accelerate",
                "scikit-learn", "statsmodels", "scipy"], check=True)

import os, json
import numpy as np
import pandas as pd
import torch
from scipy import stats
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments,
                          Trainer, set_seed)

# ---- 1. Config ---------------------------------------------
DATA_PATH = "/content/drive/MyDrive/unemployee.csv"
TEXT_COL = None     # set e.g. "text"  (None = auto-detect)
LABEL_COL = None    # set e.g. "label" (None = auto-detect)
SEEDS = [42, 123, 2024]
TEST_SIZE = 0.30

# Both pretrained models are downloaded automatically from Hugging Face
MODELS = {
    "XLM-RoBERTa":   "xlm-roberta-base",
    "AfroXLMR-base": "Davlan/afro-xlmr-base",
}
MAX_LEN, EPOCHS, LR, BATCH = 128, 3, 2e-5, 16
OUT_DIR = "/content/drive/MyDrive/significance_results"

# ---- 2. Load data ------------------------------------------
drive.mount("/content/drive")
os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_PATH)
print("Columns found:", list(df.columns), "| rows:", len(df))

if TEXT_COL is None or LABEL_COL is None:
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if TEXT_COL is None:   # text = string column with the longest average length
        TEXT_COL = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())
    if LABEL_COL is None:  # label = remaining column with fewest unique values
        others = [c for c in df.columns if c != TEXT_COL]
        LABEL_COL = min(others, key=lambda c: df[c].nunique())
print(f"Using TEXT_COL='{TEXT_COL}', LABEL_COL='{LABEL_COL}'  (edit config if wrong)")

df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
df[TEXT_COL] = df[TEXT_COL].astype(str)
le = LabelEncoder()
df["y"] = le.fit_transform(df[LABEL_COL])
num_labels = len(le.classes_)
print("Classes:", dict(enumerate(le.classes_)))
print("Class counts:\n", df[LABEL_COL].value_counts())


# ---- 3. Train + predict one model --------------------------
def train_and_predict(model_name, train_df, test_df, seed):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)

    def enc(batch):
        return tok(batch[TEXT_COL], truncation=True, max_length=MAX_LEN)

    def to_ds(d):
        d = d[[TEXT_COL, "y"]].rename(columns={"y": "labels"})
        return Dataset.from_pandas(d, preserve_index=False).map(enc, batched=True)

    tr, te = to_ds(train_df), to_ds(test_df)
    # Fresh pretrained weights each run -> true from-scratch fine-tuning
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels)

    args = TrainingArguments(
        output_dir=f"/content/tmp_{seed}", num_train_epochs=EPOCHS,
        learning_rate=LR, per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=64, weight_decay=0.01, warmup_ratio=0.1,
        seed=seed, data_seed=seed, save_strategy="no", report_to="none",
        fp16=torch.cuda.is_available(), logging_steps=50)

    trainer = Trainer(model=model, args=args, train_dataset=tr,
                      data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    preds = trainer.predict(te).predictions.argmax(axis=-1)
    del model, trainer
    torch.cuda.empty_cache()
    return preds


# ---- 4. Main loop: 3 seeds x 2 models ----------------------
results, mc_rows = [], []

for seed in SEEDS:
    # Same stratified 70/30 split for both models within a seed (paired design)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=seed, stratify=df["y"])
    y_true = test_df["y"].values
    preds = {}

    for name, path in MODELS.items():
        cache = f"{OUT_DIR}/preds_{name}_seed{seed}.npy"
        if os.path.exists(cache):                       # resume after disconnect
            print(f"[resume] loading saved predictions: {cache}")
            preds[name] = np.load(cache)
        else:
            print(f"\n=== TRAINING {name} | seed {seed} | "
                  f"train={len(train_df)} test={len(test_df)} ===")
            preds[name] = train_and_predict(path, train_df, test_df, seed)
            np.save(cache, preds[name])
        results.append({
            "seed": seed, "model": name,
            "accuracy": accuracy_score(y_true, preds[name]) * 100,
            "macro_f1": f1_score(y_true, preds[name], average="macro") * 100})
        print(f"{name} seed {seed}: acc={results[-1]['accuracy']:.2f}")

    # McNemar per seed (A = XLM-R, B = AfroXLMR)
    a_ok = preds["XLM-RoBERTa"] == y_true
    b_ok = preds["AfroXLMR-base"] == y_true
    table = [[int(np.sum(a_ok & b_ok)),  int(np.sum(a_ok & ~b_ok))],
             [int(np.sum(~a_ok & b_ok)), int(np.sum(~a_ok & ~b_ok))]]
    mc_rows.append({"seed": seed,
                    "XLMR_only_correct": table[0][1],
                    "Afro_only_correct": table[1][0],
                    "p_value": float(mcnemar(table, exact=True).pvalue)})

# ---- 5. Report ---------------------------------------------
res_df = pd.DataFrame(results)
res_df.to_csv(f"{OUT_DIR}/per_seed_results.csv", index=False)
print("\n===== PER-SEED RESULTS =====")
print(res_df.round(2).to_string(index=False))

print("\n===== MEAN ± STD over seeds (sample std, ddof=1) =====")
for m in MODELS:
    r = res_df[res_df.model == m]
    print(f"{m:15s} Acc: {r.accuracy.mean():.2f} ± {r.accuracy.std(ddof=1):.2f} | "
          f"Macro-F1: {r.macro_f1.mean():.2f} ± {r.macro_f1.std(ddof=1):.2f}")

mc_df = pd.DataFrame(mc_rows)
mc_df.to_csv(f"{OUT_DIR}/mcnemar_per_seed.csv", index=False)
print("\n===== McNEMAR'S TEST (exact, per seed) =====")
print(mc_df.to_string(index=False))
print("Bonferroni-adjusted alpha for 3 tests = 0.05/3 = 0.0167")

b_tot, c_tot = mc_df.XLMR_only_correct.sum(), mc_df.Afro_only_correct.sum()
pooled_p = mcnemar([[0, int(b_tot)], [int(c_tot), 0]], exact=True).pvalue
print(f"\nPooled discordant pairs: XLM-R-only={b_tot}, Afro-only={c_tot}, "
      f"p={pooled_p:.4g}  (indicative only: test sets overlap across seeds)")

a = res_df[res_df.model == "XLM-RoBERTa"].accuracy.values
b = res_df[res_df.model == "AfroXLMR-base"].accuracy.values
t, p = stats.ttest_rel(b, a)
print(f"Paired t-test on per-seed accuracy: t={t:.3f}, p={p:.4f} (n=3, low power)")

json.dump({"pooled_mcnemar_p": float(pooled_p), "paired_t_p": float(p)},
          open(f"{OUT_DIR}/pooled_tests.json", "w"), indent=2)
print(f"\nAll results saved in {OUT_DIR}")
