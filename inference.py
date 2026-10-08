"""
inference.py — MindCare dual-head inference module.

Loads the frozen artefacts produced by `mindcare_train.ipynb`:
    expanded_9disorder_model.h5   Keras model (custom TokenLevelAttention layer)
    expanded_tokenizer.pkl        plain-dict vocabulary  {word_index, oov_token, num_words, max_len}
    expanded_labels.pkl           label spaces + metadata (trained / untrained disorders, MAX_LEN, ...)

Public API
    predict_pipeline(text)  -> dict   (single text, full computation trace)
    predict_batch(texts)    -> list[dict]
    load_artifacts()        -> loads once (thread-safe); called lazily by the functions above
    warm_up()               -> load + run one dummy forward pass (use at server start-up)

Artefact folder resolution (first match wins):
    load_artifacts(model_dir=...) argument  >  $MINDCARE_MODEL_DIR  >  ./models  >  .  >  <this file>/models  >  <this file>/

Returned probabilities are floats in [0, 1]; multiply by 100 for percentages.
"""
import logging
import os
import re
import threading
from typing import Dict, List, Optional

from collections import Counter  

import joblib
import keras
import numpy as np
import tensorflow as tf
from keras import layers

logger = logging.getLogger("mindcare.inference")

MODEL_FILE     = "expanded_9disorder_model.h5"
TOKENIZER_FILE = "expanded_tokenizer.pkl"
LABELS_FILE    = "expanded_labels.pkl"


# ──────────────────────────────────────────────────────────────────────────────
# Custom layer — MUST stay identical to the class in mindcare_train.ipynb
# ──────────────────────────────────────────────────────────────────────────────
class TokenLevelAttention(layers.Layer):
    """Scaled dot-product self-attention across tokens.
    Input: [embeddings (B, T, D), token_ids (B, T)] -> Output: (B, T, D)."""

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

# ──────────────────────────────────────────────────────────────────────────────
# Text utilities — MUST stay identical to the notebook
# ──────────────────────────────────────────────────────────────────────────────
_URL_RE     = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")

# 🌟 MODIFICATION 1: Strict multi-word structural replacements to defend against temporal false-positives
TEMPORAL_INVERSIONS = {
    "used to love": "no longer love",
    "used to enjoy": "no longer enjoy",
    "used to like": "no longer like",
    "used to feel": "no longer feel"
}

def clean_text(text) -> str:
    """Lower-case, strip URLs/@mentions, normalize temporal inversions,
    collapse contractions, and apply strict dictionary fallback tokens.
    Preserves Urdu unicode characters."""
    from modules.text.urdu_sentiment import is_urdu_text, normalize_urdu_script
    if is_urdu_text(str(text)):
        return normalize_urdu_script(str(text))

    text = str(text).lower().replace("\u2019", "'")
    
    # 🚀 CRITICAL FIX: Intercept and substitute past-tense phrasing before token division occurs
    for custom_phrase, safe_replacement in TEMPORAL_INVERSIONS.items():
        if custom_phrase in text:
            text = text.replace(custom_phrase, safe_replacement)
            
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9\s!?]", " ", text)
    text = re.sub(r"([!?])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode_texts(texts, word_index, max_len):
    seqs = [[word_index.get(tok, 1) for tok in t.split()] for t in texts]
    return keras.utils.pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post", value=0)


# 🌟 MODIFICATION 2: Synchronized Emotion Lexicon to mirror lookback penalizations during inference
EMOTIONS  = ["normal", "happy", "sad", "angry", "frustrated", "anxious"]
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
NEGATORS = {"not","no","never","dont","cant","isnt","wasnt","aint","without","hardly","barely","wont","didnt","doesnt","nothing","none"}

def emotion_hits(clean: str) -> Counter:
    toks, hits = clean.split(), Counter()
    for i, tok in enumerate(toks):
        for emo, vocab in EMOTION_LEXICON.items():
            lookback_window = set(toks[max(0, i - 4):i])
            if tok in vocab:
                if NEGATORS & lookback_window:
                    if emo == 'happy':
                        hits['sad'] += 1.5  
                else:
                    hits[emo] += 1
    for emo, phrases in EMOTION_PHRASES.items():
        hits[emo] += sum(p in clean for p in phrases)
    return hits

def weak_emotion(clean: str, prior: str) -> str:
    # 🌟 MODIFICATION 3: Explicit protection fallback matching train.py precisely
    ghost_anchors = {"ghost", "staring blankly", "nothing sparks", "flatness", "pointless", "hollow", "secluded", "isolated", "burnout"}
    if any(anchor in clean for anchor in ghost_anchors):
        return "sad"
        
    hits = emotion_hits(clean)
    if not hits:
        return prior
    top = max(hits.values())
    tied = [e for e in EMOTIONS if hits.get(e, 0) == top]
    return prior if prior in tied else tied[0]


# Keyword safety net (independent of the model — the model has no self-harm class)
_SELF_HARM_RE = re.compile(
    r"\b(kill(?:ing)? myself|end(?:ing)? my (?:own )?life|take my (?:own )?life|want(?:ed)? to die|wanna die|"
    r"suicid\w*|better off dead|dont want to live|no reason to live|end it all|"
    r"hurt(?:ing)? myself|harm(?:ing)? myself|self harm|cutting myself)\b"
)
SAFETY_NOTICE = (
    "If you are thinking about harming yourself or ending your life, please reach out right now: contact your local "
    "emergency number or a crisis line (Pakistan: 0311-7786264 · elsewhere: findahelpline.com). You are not alone, "
    "and talking to someone you trust can help."
)

# ──────────────────────────────────────────────────────────────────────────────
# Artefact loading (lazy, thread-safe, cached)
# ──────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_predict_lock = threading.Lock()
_STATE: Optional[dict] = None


def _candidate_dirs(model_dir: Optional[str]) -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [model_dir, os.environ.get("MINDCARE_MODEL_DIR"), "models", ".",
             os.path.join(here, "models"), here]
    seen, out = set(), []
    for c in cands:
        if c and os.path.abspath(c) not in seen:
            seen.add(os.path.abspath(c)); out.append(c)
    return out


def _find_dir(model_dir: Optional[str]) -> str:
    for d in _candidate_dirs(model_dir):
        if all(os.path.isfile(os.path.join(d, f)) for f in (MODEL_FILE, TOKENIZER_FILE, LABELS_FILE)):
            return os.path.abspath(d)
    searched = ", ".join(os.path.abspath(d) for d in _candidate_dirs(model_dir))
    raise FileNotFoundError(
        f"MindCare artefacts not found. Need {MODEL_FILE}, {TOKENIZER_FILE}, {LABELS_FILE} in one folder. "
        f"Searched: {searched}. Train with mindcare_train.ipynb and copy the models/ folder here, "
        f"or set MINDCARE_MODEL_DIR."
    )


def load_artifacts(model_dir: Optional[str] = None, force_reload: bool = False) -> dict:
    global _STATE
    if _STATE is not None and not force_reload:
        return _STATE
    with _lock:
        if _STATE is not None and not force_reload:
            return _STATE
        folder = _find_dir(model_dir)
        tok    = joblib.load(os.path.join(folder, TOKENIZER_FILE))
        labels = joblib.load(os.path.join(folder, LABELS_FILE))
        model  = keras.models.load_model(
            os.path.join(folder, MODEL_FILE),
            custom_objects={"TokenLevelAttention": TokenLevelAttention},
            compile=False,
        )
        saved_tf = str(labels.get("tensorflow_version", ""))
        if saved_tf and saved_tf.split(".")[:2] != tf.__version__.split(".")[:2]:
            logger.warning("Model trained on TF %s but running TF %s — pin the same version if loading misbehaves.",
                           saved_tf, tf.__version__)
        _STATE = {
            "model": model,
            "word_index": tok["word_index"],
            "max_len": int(labels.get("max_len", tok.get("max_len", 50))),
            "emotions": list(labels["emotions"]),
            "disorders": list(labels["disorders"]),
            "trained_disorders": list(labels.get("trained_disorders", labels["disorders"])),
            "untrained_disorders": list(labels.get("untrained_disorders", [])),
            "folder": folder,
            "meta": {k: labels.get(k) for k in ("emotion_test_accuracy", "disorder_test_auc",
                                                "tensorflow_version", "keras_version")},
        }
        logger.info("MindCare artefacts loaded from %s (untrained heads: %s)", folder, _STATE["untrained_disorders"])
        return _STATE


def _unpack(pred):
    """Keras returns a dict or a list depending on version / load path — accept both."""
    if isinstance(pred, dict):
        return pred["emotion_head"], pred["disorder_head"]
    return pred[0], pred[1]


def warm_up() -> dict:
    """Load artefacts and run one dummy forward pass (call at server start)."""
    predict_pipeline("warm up check")
    return load_artifacts()


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────
def _trace(text_input: str, clean: str, ids_row: np.ndarray,
           emo: np.ndarray, dis: np.ndarray, st: dict) -> Dict:
    tokens = clean.split()
    n_real = int(min(len(tokens), st["max_len"]))
    words = [t for t in tokens if t not in ("!", "?")]

    # ── Emotion head ─────────────────────────────────────────────────────
        # ── Emotion head ─────────────────────────────────────────────────────
    emo_idx = int(np.argmax(emo))
    model_top = st["emotions"][emo_idx]
    model_conf = float(emo[emo_idx])

    # Lexicon override: when the neural model is not confidently sure
    # (< 0.70), ask the keyword-lexicon layer what it sees. This catches
    # cases the model misses on short or unusual phrasing, e.g.
    # "I used to love X but now nothing interests me" (anhedonia) where
    # the model keys on the word "love" without weighing the negation.
    OVERRIDE_BELOW = 0.70
    if model_conf < OVERRIDE_BELOW:
        lexicon_emotion = weak_emotion(clean, prior=model_top)
        if lexicon_emotion != model_top:
            emotion_name = lexicon_emotion
            # Report the model's confidence for the LEXICON-chosen class,
            # not for the class the model originally chose.
            emotion_conf = float(emo[st["emotions"].index(lexicon_emotion)])
            overridden = True
        else:
            emotion_name = model_top
            emotion_conf = model_conf
            overridden = False
    else:
        emotion_name = model_top
        emotion_conf = model_conf
        overridden = False
    # ── Disorder head ────────────────────────────────────────────────────
    # Two-tier design:
    #   * `disorder_probabilities` shows `None` for untrained heads, so a
    #     frontend can never accidentally read noise as signal.
    #   * `top_disorder` is only emitted above CONFIDENCE_FLOOR. Below that
    #     we return None and the recommendations layer falls back to
    #     emotion-based supportive language.
    CONFIDENCE_FLOOR = 0.75

    disorder_probabilities = {}
    trained_only = {}
    for i, name in enumerate(st["disorders"]):
        if name in st["trained_disorders"]:
            p = float(dis[i])
            disorder_probabilities[name] = p
            trained_only[name] = p
        else:
            disorder_probabilities[name] = None   # untrained — do not interpret

    if trained_only:
        top_name = max(trained_only, key=trained_only.get)
        top_prob = trained_only[top_name]
        top_disorder = (
            {"name": top_name, "probability": top_prob}
            if top_prob >= CONFIDENCE_FLOOR
            else None
        )
    else:
        top_disorder = None

    # ── Safety check ─────────────────────────────────────────────────────
    # Check BOTH the cleaned text AND the original input. Cleaning can
    # strip punctuation and break regex anchors ("kill myself!" vs
    # "kill myself"), so the raw input is checked too.
    safety_notice = None
    if (_SELF_HARM_RE.search(clean.lower())
            or _SELF_HARM_RE.search(str(text_input).lower())):
        safety_notice = SAFETY_NOTICE

    return {
        "input_text":             text_input,
        "clean_text":             clean,
        "clean_tokens":           tokens,
        "token_ids":              [int(i) for i in ids_row[:n_real]],
        "padded_length":          int(st["max_len"]),
        "truncated":              len(tokens) > st["max_len"],

        # Emotion head
        "selected_emotion":       emotion_name,
        "emotion_confidence":     emotion_conf,
        "emotion_probabilities":  {e: float(emo[i]) for i, e in enumerate(st["emotions"])},
        "emotion_overridden":     overridden, 
        # Disorder head
        "disorder_probabilities": disorder_probabilities,
        "trained_disorders":      list(st["trained_disorders"]),
        "untrained_disorders":    list(st["untrained_disorders"]),
        "top_disorder":           top_disorder,

        # Safety
        "safety_notice":          safety_notice,
    }

def predict_batch(texts: List[str]) -> List[Dict]:
    """Batch version of predict_pipeline. Handles both English and Urdu/Roman-Urdu text seamlessly."""
    from modules.text.urdu_sentiment import is_urdu_text, analyze_urdu_sentiment

    st = load_artifacts()
    results = [None] * len(texts)
    en_indices = []
    en_cleaned = []

    for idx, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            raise ValueError("Every input must be a non-empty string.")
        
        # Branch to Urdu sentiment engine if Urdu script or Roman Urdu is detected
        if is_urdu_text(t):
            results[idx] = analyze_urdu_sentiment(t)
        else:
            c = clean_text(t)
            if not re.search(r"[a-z0-9]", c):
                # Fallback check: could it be unmapped Urdu characters?
                if any(ord(char) >= 0x0600 for char in t):
                    results[idx] = analyze_urdu_sentiment(t)
                    continue
                raise ValueError("Input contains no usable words after cleaning.")
            en_indices.append(idx)
            en_cleaned.append(c)

    # Process English batch through dual-head attention model
    if en_cleaned:
        ids = encode_texts(en_cleaned, st["word_index"], st["max_len"]).astype("int32")  # (B, max_len)
        with _predict_lock:
            emo, dis = _unpack(st["model"](ids, training=False))
        emo, dis = np.asarray(emo), np.asarray(dis)
        for i, idx in enumerate(en_indices):
            results[idx] = _trace(texts[idx], en_cleaned[i], ids[i], emo[i], dis[i], st)

    return results


def predict_pipeline(text_input: str) -> Dict:
    """Clean -> tokenise (saved vocabulary) -> pad to a (1, MAX_LEN) batch -> multi-task forward pass -> trace dict."""
    return predict_batch([text_input])[0]


if __name__ == "__main__":
    import json, sys
    sample = " ".join(sys.argv[1:]) or "I keep checking the stove and locking the door again and again."
    print(json.dumps(predict_pipeline(sample), indent=2))