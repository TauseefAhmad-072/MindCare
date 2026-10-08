"""
train.py — MindCare training pipeline as a plain script (same cells as mindcare_train.ipynb).

    python train.py

Writes models/expanded_9disorder_model.h5, models/expanded_tokenizer.pkl, models/expanded_labels.pkl
(+ mindcare_models.zip). Needs internet on first run to download the two Hugging Face datasets.
Optional: MINDCARE_EXTRA_CLINICAL=path/to/file.csv  adds anxiety/bipolar/BPD/schizophrenia posts (see README).
"""

# # MindCare — Dual-Head Token-Level Self-Attention (Training Pipeline)
#
# **Replaces the legacy SBERT / LR / LSTM / BiLSTM pipeline.** One model, two heads:
#
# | Head | Output | Activation | Classes |
# |---|---|---|---|
# | `emotion_head` | single winning emotion | `softmax` | normal, happy, sad, angry, frustrated, anxious |
# | `disorder_head` | 9 independent probabilities | `sigmoid` | adhd, anxiety_disorder, aspergers, bipolar, bpd, depression, ocd, ptsd, schizophrenia |
#
# **Data (kept in three separate loaders, merged only in section 5):**
# - **A. Reddit clinical** — `solomonk/reddit_mental_health_posts`
# - **B. Optional extra clinical file** — local CSV/Parquet/JSONL for disorders missing from A
# - **C. Twitter controls** — `gxb912/large-twitter-tweets-sentiment` (healthy normal/happy, plus non-clinical negative-emotion controls)

# ## 1. Imports & configuration

import os, re, json, random, shutil, warnings
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import keras
from keras import layers, Model
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ── Architecture / pipeline constants (do not change after training: inference reads them from the saved labels file)
MAX_WORDS   = 20000      # Embedding(20000, 128)
EMBED_DIM   = 128
KEY_DIM     = 32
MAX_LEN     = 50         # strict sequence length
SAMPLES_PER_CLASS = 2500 # balanced sampling threshold
MIN_TRAIN_SAMPLES = 200  # a disorder with fewer positives is flagged "untrained"
EPOCHS      = 5
BATCH_SIZE  = 64

# ── Label spaces
EMOTIONS  = ["normal", "happy", "sad", "angry", "frustrated", "anxious"]
DISORDERS = ["adhd", "anxiety_disorder", "aspergers", "bipolar", "bpd",
             "depression", "ocd", "ptsd", "schizophrenia"]

# ── Data sources
REDDIT_DATASET  = "solomonk/reddit_mental_health_posts"
TWITTER_DATASET = "gxb912/large-twitter-tweets-sentiment"
EXTRA_CLINICAL_PATH = os.environ.get("MINDCARE_EXTRA_CLINICAL")   
USE_NEGATIVE_TWEETS_FOR_EMOTION = True  

# ── Output artefacts
ARTIFACT_DIR   = "models"
MODEL_FILE     = "expanded_9disorder_model.h5"
TOKENIZER_FILE = "expanded_tokenizer.pkl"
LABELS_FILE    = "expanded_labels.pkl"

print("TensorFlow", tf.__version__, "| Keras", keras.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU") or "none (training will run on CPU)")

# ## 2. Text utilities (must stay identical to `inference.py`)

_URL_RE     = re.compile(r"https?://\S+|www.\S+")
_MENTION_RE = re.compile(r"@\w+")

def clean_text(text) -> str:
    """Lower-case, strip URLs/@mentions, collapse contractions (don't -> dont),
    keep ! and ? as their own tokens, replace every other symbol with a space."""
    text = str(text).lower().replace("\u2019", "'")
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9\s!?]", " ", text)
    text = re.sub(r"([!?])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()

def build_vocab(texts, max_words=MAX_WORDS):
    """word_index: 0 = PAD, 1 = <UNK>, 2.. = most frequent tokens. Max index = max_words - 1."""
    counts = Counter(tok for t in texts for tok in t.split())
    word_index = {"<UNK>": 1}
    for i, (w, _) in enumerate(counts.most_common(max_words - 2), start=2):
        word_index[w] = i
    return word_index

def encode_texts(texts, word_index, max_len=MAX_LEN):
    seqs = [[word_index.get(tok, 1) for tok in t.split()] for t in texts]
    return keras.utils.pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post", value=0)

# ── Weak emotion labeller (clinical posts have no emotion labels; this + a per-disorder prior supplies them)
EMOTION_LEXICON = {
    "sad":        {"sad","cry","crying","cried","tears","hopeless","empty","lonely","alone","depressed","miserable",
                   "grief","heartbroken","worthless","numb","hollow","unhappy","hurts","hurting","gloomy","sorrow"},
    "angry":      {"angry","anger","furious","rage","mad","hate","hated","pissed","livid","enraged","hostile",
                   "resent","fuming","irate","outraged"},
    "frustrated": {"frustrated","frustrating","frustration","annoyed","annoying","irritated","irritating","stuck",
                   "ugh","exhausted","pointless","useless","struggling","struggle","tired"},
    "anxious":    {"anxious","anxiety","worried","worry","panic","nervous","scared","afraid","fear","terrified",
                   "dread","paranoid","overthinking","uneasy","stressed","stress","overwhelmed","racing"},
    "happy":      {"happy","glad","excited","love","loved","great","amazing","awesome","wonderful","joy",
                   "grateful","proud","thrilled","enjoy","enjoyed","fun","lol","haha","yay","blessed"},
}
EMOTION_PHRASES = {"frustrated": ["fed up", "sick of", "tired of"], "anxious": ["on edge", "what if"], "sad": ["feel down", "give up"]}

# 🌟 MODIFICATION 1: Expanded to catch multi-word absolute structural negations
NEGATORS = {"not","no","never","dont","cant","isnt","wasnt","aint","without","hardly","barely","wont","didnt","doesnt","nothing","none"}

def emotion_hits(clean: str) -> Counter:
    toks, hits = clean.split(), Counter()
    for i, tok in enumerate(toks):
        for emo, vocab in EMOTION_LEXICON.items():
            # 🌟 MODIFICATION 2: Expanded back to 4 words window to map "nothing brings me joy"
            lookback_window = set(toks[max(0, i - 4):i])
            
            if tok in vocab:
                if NEGATORS & lookback_window:
                    # 🌟 MODIFICATION 3: Invert positive parameters to sad to fix false positive happy leaks
                    if emo == 'happy':
                        hits['sad'] += 1.5
                else:
                    hits[emo] += 1
    for emo, phrases in EMOTION_PHRASES.items():
        hits[emo] += sum(p in clean for p in phrases)
    return hits

def weak_emotion(clean: str, prior: str) -> str:
    # 🌟 MODIFICATION 4: Strict fallback anchors to defend against apathetic burnout classification errors
    flatness_anchors = {"pointless", "hollow", "flatness", "numb", "secluded", "isolated", "burnout"}
    if any(anchor in clean for anchor in flatness_anchors):
        return "sad"

    hits = emotion_hits(clean)
    if not hits:
        return prior
    top = max(hits.values())
    tied = [e for e in EMOTIONS if hits.get(e, 0) == top]
    return prior if prior in tied else tied[0]

print("clean_text demo :", clean_text("I DON'T feel okay!! https://x.co @friend — can't sleep?"))
print("weak_emotion demo:", weak_emotion(clean_text("I am not happy, I feel so hopeless and empty"), "normal"))

# ## 3. Custom `TokenLevelAttention` layer (scratch-built, `key_dim=32`)

class TokenLevelAttention(layers.Layer):
    def __init__(self, key_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = int(key_dim)

    def build(self, input_shape):
        d = int(input_shape[0][-1])
        self.wq = self.add_weight(name="wq", shape=(d, self.key_dim), initializer="glorot_uniform")
        self.wk = self.add_weight(name="wk", shape=(d, self.key_dim), initializer="glorot_uniform")
        self.wv = self.add_weight(name="wv", shape=(d, d),            initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs):
        x, ids = inputs
        valid = tf.not_equal(ids, 0)
        valid = tf.logical_or(valid, tf.logical_not(tf.reduce_any(valid, axis=1, keepdims=True)))
        q, k, v = tf.matmul(x, self.wq), tf.matmul(x, self.wk), tf.matmul(x, self.wv)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, x.dtype))
        scores = tf.where(valid[:, None, :], scores, tf.constant(-1e9, scores.dtype))
        out = tf.matmul(tf.nn.softmax(scores, axis=-1), v)
        return tf.where(valid[:, :, None], out, tf.constant(-1e9, out.dtype))

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"key_dim": self.key_dim})
        return cfg

print("TokenLevelAttention ready.")

# ## 4. Datasets — loaded and prepared separately

SUBREDDIT_TO_DISORDER = {
    "adhd": "adhd", "anxiety": "anxiety_disorder", "anxious": "anxiety_disorder",
    "aspergers": "aspergers", "autism": "aspergers", "bipolar": "bipolar",
    "bpd": "bpd", "borderlinepersonalitydisorder": "bpd",
    "depression": "depression", "ocd": "ocd", "ptsd": "ptsd", "schizophrenia": "schizophrenia",
}
DISORDER_EMOTION_PRIOR = {
    "adhd": "frustrated", "anxiety_disorder": "anxious", "aspergers": "normal", "bipolar": "frustrated",
    "bpd": "sad", "depression": "sad", "ocd": "anxious", "ptsd": "anxious", "schizophrenia": "anxious",
}
OUT_COLS = ["text", "emotion", "disorder", "stratum", "source"]

def prepare_clinical_frame(
    raw: pd.DataFrame, source: str, only_disorders=None
) -> pd.DataFrame:
    df = raw.copy()
    sub_col = next(
        (c for c in ("subreddit", "label", "category") if c in df.columns), None
    )
    body_col = next(
        (c for c in ("body", "selftext", "text", "post") if c in df.columns), None
    )
    if sub_col is None or body_col is None:
        raise ValueError(
            f"{source}: need a subreddit column and a body/selftext column. "
            f"Found: {list(df.columns)}"
        )

    df["disorder"] = (
        df[sub_col].astype(str).str.lower().str.strip().map(SUBREDDIT_TO_DISORDER)
    )
    df = df.dropna(subset=["disorder"])
    if only_disorders is not None:
        df = df[df["disorder"].isin(only_disorders)]
    if df.empty:
        return pd.DataFrame(columns=OUT_COLS)

    df = pd.concat([
        group.sample(n=min(len(group), SAMPLES_PER_CLASS * 3), random_state=SEED)
        for _, group in df.groupby("disorder")
    ])

    body = df[body_col].fillna("").astype(str).replace(
        {"[removed]": "", "[deleted]": ""}
    )
    title = df["title"].fillna("").astype(str) if "title" in df.columns else ""
    df["text"] = (title + " " + body).apply(clean_text)
    df = df[df["text"].str.split().str.len() >= 5].drop_duplicates("text")
    df = pd.concat([
        group.sample(n=min(len(group), SAMPLES_PER_CLASS), random_state=SEED)
        for _, group in df.groupby("disorder")
    ])

    df["emotion"] = [
        weak_emotion(text, DISORDER_EMOTION_PRIOR[disorder])
        for text, disorder in zip(df["text"], df["disorder"])
    ]
    df["stratum"] = df["disorder"]
    df["source"] = source
    return df[OUT_COLS].reset_index(drop=True)
def load_reddit_clinical() -> pd.DataFrame:
    ds = load_dataset(REDDIT_DATASET)
    raw = ds["train"].to_pandas()
    print("Reddit subreddits found:", sorted(raw["subreddit"].astype(str).unique()))
    return prepare_clinical_frame(raw, "reddit:solomonk")

reddit_df = load_reddit_clinical()
print(f"\nReddit clinical frame: {len(reddit_df)} rows")
print(reddit_df.groupby("disorder").size().to_string())

# 4B. Optional extra clinical file**

missing_disorders = [d for d in DISORDERS if d not in set(reddit_df["disorder"])]
print("Disorders with NO rows in the Reddit dataset:", missing_disorders or "none")

def load_extra_clinical(path, missing) -> pd.DataFrame:
    if not path or not missing:
        return pd.DataFrame(columns=OUT_COLS)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        raw = pd.read_parquet(path)
    elif ext in (".json", ".jsonl"):
        raw = pd.read_json(path, lines=True)
    else:
        raw = pd.read_csv(path)

    return prepare_clinical_frame(
        raw,
        f"extra:{os.path.basename(path)}",
        only_disorders=missing,
    )
extra_df = load_extra_clinical(EXTRA_CLINICAL_PATH, missing_disorders)
print(f"Extra clinical frame: {len(extra_df)} rows")
if len(extra_df):
    print(extra_df.groupby("disorder").size().to_string())
elif missing_disorders:
    print("→ Those disorder heads will be flagged UNTRAINED. Set EXTRA_CLINICAL_PATH to train them.")

def load_twitter_controls(per_polarity=30000) -> pd.DataFrame:
    ds = load_dataset(TWITTER_DATASET)
    raw = ds["train"].to_pandas()
    sent_col = "sentiment" if "sentiment" in raw.columns else "label"
    text_col = next(
        c for c in raw.columns
        if c != sent_col and pd.api.types.is_string_dtype(raw[c])
    )
    raw = raw[[text_col, sent_col]].rename(
        columns={text_col: "raw", sent_col: "sentiment"}
    )
    raw["sentiment"] = pd.to_numeric(raw["sentiment"], errors="coerce")
    raw = raw.dropna()

    rows = []
    for polarity, grp in raw.groupby("sentiment"):
        grp = grp.sample(n=min(len(grp), per_polarity), random_state=SEED)
        for text in grp["raw"].apply(clean_text):
            if len(text.split()) < 3:
                continue
            hits = emotion_hits(text)
            if polarity == 1:
                emo = "happy" if hits.get("happy", 0) > 0 else "normal"
            elif USE_NEGATIVE_TWEETS_FOR_EMOTION:
                neg = {
                    e: hits.get(e, 0)
                    for e in ("sad", "angry", "frustrated", "anxious")
                    if hits.get(e, 0) > 0
                }
                if not neg:
                    continue
                emo = max(neg, key=neg.get)
            else:
                continue
            rows.append({
                "text": text,
                "emotion": emo,
                "disorder": None,
                "stratum": f"control_{emo}",
                "source": "twitter:gxb912",
            })
    return (
        pd.DataFrame(rows, columns=OUT_COLS)
        .drop_duplicates("text")
        .reset_index(drop=True)
    )
twitter_df = load_twitter_controls()
print(f"Twitter control frame: {len(twitter_df)} rows")
print(twitter_df.groupby("emotion").size().to_string())

# 5. Merge & balance**

all_df = pd.concat([reddit_df, extra_df, twitter_df], ignore_index=True).drop_duplicates("text")

counts = all_df["stratum"].value_counts()
too_small = counts[counts < 30].index.tolist()
if too_small:
    print("Dropping strata with <30 rows:", too_small)
    all_df = all_df[~all_df["stratum"].isin(too_small)]

master_df = pd.concat(
    [g.sample(n=min(len(g), SAMPLES_PER_CLASS), random_state=SEED) for _, g in all_df.groupby("stratum")]
    ).reset_index(drop=True)

dis2id = {d: i for i, d in enumerate(DISORDERS)}
y_disorders_all = np.zeros((len(master_df), len(DISORDERS)), dtype="float32")
for r, d in enumerate(master_df["disorder"]):
    if d is not None and d == d and d in dis2id:
        y_disorders_all[r, dis2id[d]] = 1.0
emo2id = {e: i for i, e in enumerate(EMOTIONS)}
y_emotion_all = master_df["emotion"].map(emo2id).to_numpy(dtype="int64")

pos_counts = {d: int(y_disorders_all[:, i].sum()) for d, i in dis2id.items()}
TRAINED_DISORDERS   = [d for d in DISORDERS if pos_counts[d] >= MIN_TRAIN_SAMPLES]
UNTRAINED_DISORDERS = [d for d in DISORDERS if pos_counts[d] <  MIN_TRAIN_SAMPLES]

print(f"Master dataset: {len(master_df)} rows\n")
print("Stratum counts (cap = %d):" % SAMPLES_PER_CLASS)
print(master_df["stratum"].value_counts().to_string())
print("\nEmotion distribution:")
print(master_df["emotion"].value_counts().reindex(EMOTIONS).fillna(0).astype(int).to_string())
print("\nDisorder positives:", pos_counts)

# 6. Tokenise, pad, split**

row_ids = np.arange(len(master_df))
train_ids, temp_ids = train_test_split(row_ids, test_size=0.20, random_state=SEED, stratify=master_df["stratum"])
val_ids, test_ids   = train_test_split(temp_ids, test_size=0.50, random_state=SEED,
stratify=master_df["stratum"].iloc[temp_ids])

word_index = build_vocab(master_df["text"].iloc[train_ids])
X_padded   = encode_texts(master_df["text"], word_index)
assert X_padded.shape == (len(master_df), MAX_LEN) and X_padded.max() < MAX_WORDS

rng = np.random.default_rng(SEED)
for ids in (train_ids, val_ids, test_ids):
    rng.shuffle(ids)

X_train, X_val, X_test = X_padded[train_ids], X_padded[val_ids], X_padded[test_ids]
ye_train, ye_val, ye_test = y_emotion_all[train_ids], y_emotion_all[val_ids], y_emotion_all[test_ids]
yd_train, yd_val, yd_test = y_disorders_all[train_ids], y_disorders_all[val_ids], y_disorders_all[test_ids]

w_emotion = compute_sample_weight("balanced", ye_train).astype("float32")
w_disorder = np.ones(len(X_train), dtype="float32")

# 7. Dual-head model**

def build_model():
    inputs  = layers.Input(shape=(MAX_LEN,), dtype="int32", name="text_input")
    emb     = layers.Embedding(MAX_WORDS, EMBED_DIM, name="token_embedding")(inputs)
    attn    = TokenLevelAttention(key_dim=KEY_DIM, name="token_attention")([emb, inputs])
    pooled  = layers.GlobalMaxPooling1D(name="token_max_pool")(attn)
    shared  = layers.Dropout(0.3, name="shared_dropout")(pooled)

    emo = layers.Dense(64, activation="relu", name="emotion_dense")(shared)
    emotion_out = layers.Dense(len(EMOTIONS), activation="softmax", name="emotion_head")(emo)

    dis = layers.Dense(64, activation="relu", name="disorder_dense")(shared)
    disorder_out = layers.Dense(len(DISORDERS), activation="sigmoid", name="disorder_head")(dis)

    model = Model(inputs, {"emotion_head": emotion_out, "disorder_head": disorder_out}, name="mindcare_dual_head")

    # 🌟 MODIFICATION 5: Add strategic class loss weights so neither gradient head matches dominant bias parameters
    model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss={"emotion_head": "sparse_categorical_crossentropy", "disorder_head": "binary_crossentropy"},
    loss_weights={"emotion_head": 0.5, "disorder_head": 1.0},
    metrics={"emotion_head": ["accuracy"], "disorder_head": ["binary_accuracy"]},
    )
    return model

model = build_model()

# 8. Train (5 epochs)**

# 🌟 MODIFICATION 6: Shift early stopping to patience=1 with restore weights to lock onto Epoch 2 parameters cleanly

history = model.fit(
X_train, {"emotion_head": ye_train, "disorder_head": yd_train},
sample_weight={"emotion_head": w_emotion, "disorder_head": w_disorder},
validation_data=(X_val, {"emotion_head": ye_val, "disorder_head": yd_val}),
epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2,
callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)],
)

# 9. Evaluate on the held-out test split**

def unpack(pred):
    return (pred["emotion_head"], pred["disorder_head"]) if isinstance(pred, dict) else (pred[0], pred[1])

emo_prob, dis_prob = unpack(model.predict(X_test, batch_size=256, verbose=0))
emo_pred = emo_prob.argmax(1)

print("=== Emotion head ===")
print(f"accuracy {accuracy_score(ye_test, emo_pred):.4f}")

# 10. Freeze & serialise artefacts**

os.makedirs(ARTIFACT_DIR, exist_ok=True)
model_path, tok_path, labels_path = (os.path.join(ARTIFACT_DIR, f) for f in (MODEL_FILE, TOKENIZER_FILE, LABELS_FILE))
model.save(model_path, include_optimizer=False)

joblib.dump({"word_index": word_index, "oov_token": "", "num_words": MAX_WORDS, "max_len": MAX_LEN}, tok_path)

joblib.dump({
    "emotions": EMOTIONS,
    "disorders": DISORDERS,
    "trained_disorders": TRAINED_DISORDERS,
    "untrained_disorders": UNTRAINED_DISORDERS,
    "max_len": MAX_LEN,
    "max_words": MAX_WORDS,
    "disorder_positive_counts": pos_counts,
    "emotion_test_accuracy": float(accuracy_score(ye_test, emo_pred)),
    "tensorflow_version": tf.__version__,
    "keras_version": keras.__version__,
}, labels_path)

shutil.make_archive("mindcare_models", "zip", ARTIFACT_DIR)
print("\nCreated mindcare_models.zip successfully via fixed inversion logic loops.")
