"""
modules/audio_emotion/urdu.py — Urdu speech layer.

Adds four capabilities on top of the existing SER classifier:

  1. Urdu script normalization  — canonicalizes ی/ي, ک/ك, ہ/ه, strips diacritics
  2. Crisis detection          — three regexes: English, Urdu script, Roman Urdu
  3. Whisper ASR (ur)          — tries faster-whisper → openai-whisper → transformers
  4. Prosody extraction        — F0 contour, jitter, MFCC, spectral, rhythm

Every import is lazy. If Whisper or librosa isn't installed, the layer
degrades to a clear "unavailable" status without crashing the endpoint.

Public:
    normalize_urdu(text) -> str
    safety_hit(text) -> bool
    transcribe(waveform, sr) -> dict
    extract_prosody(waveform, sr, word_count=None) -> (vector, features)
    enrich_audio_result(waveform, sr) -> dict     # convenience wrapper
"""
from __future__ import annotations

import logging
import math
import re
import threading
import unicodedata
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("mindcare.audio_emotion.urdu")

TARGET_SR = 16000


# ═══════════════════════════════════════════════════════════════════════════
# 1. Urdu script normalization
# ═══════════════════════════════════════════════════════════════════════════

_URDU_VARIANTS = {
    "ي": "ی", "ى": "ی",       # Arabic yeh → Urdu yeh
    "ك": "ک",                 # Arabic kaf → Urdu kaf
    "ه": "ہ", "ۀ": "ہ", "ة": "ہ",
    "أ": "ا", "إ": "ا",
    "٠": "۰", "١": "۱", "٢": "۲", "٣": "۳", "٤": "۴",
    "٥": "۵", "٦": "۶", "٧": "۷", "٨": "۸", "٩": "۹",
}

# diacritics, tatweel, zero-width joiners
_STRIP_RE = re.compile(r"[\u064b-\u065f\u0670\u0640\u200b-\u200f\ufeff]")


def normalize_urdu(text: str) -> str:
    """Canonicalize Urdu script variants so lexicon matching is stable."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(_URDU_VARIANTS.get(c, c) for c in text)
    text = _STRIP_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Crisis detection — English, Urdu script, Roman Urdu
# ═══════════════════════════════════════════════════════════════════════════

_EN_CRISIS = re.compile(
    r"\b(kill(?:ing)?\s+myself|end(?:ing)?\s+my\s+(?:own\s+)?life|"
    r"take\s+my\s+(?:own\s+)?life|want(?:ed)?\s+to\s+die|wanna\s+die|"
    r"suicid\w*|better\s+off\s+dead|don'?t\s+want\s+to\s+live|"
    r"no\s+reason\s+to\s+live|end\s+it\s+all|"
    r"hurt(?:ing)?\s+myself|harm(?:ing)?\s+myself|self[\s-]?harm|"
    r"cutting\s+myself)\b",
    re.IGNORECASE,
)

_URDU_CRISIS = re.compile(
    r"(خود\s*کشی"
    r"|اپنی\s+جان\s+(?:لے|دے|دینا)"
    r"|مرنا\s+چاہت[ای]\s+ہوں"
    r"|مر\s*جانا\s+چاہت[ای]"
    r"|جینا\s+نہیں\s+چاہت[ای]"
    r"|زندہ\s+نہیں\s+رہنا\s+چاہت[ای]"
    r"|زندگی\s+ختم\s+کر"
    r"|اپنے\s+آپ\s+کو\s+(?:ختم|نقصان|زخمی)"
    r"|خود\s+کو\s+(?:ختم|نقصان))"
)

_ROMAN_URDU_CRISIS = re.compile(
    r"\b("
    r"khud[\s-]?kushi|khudkushi"
    r"|marna\s+chaht[ai]\s+ho?n?"
    r"|mar\s+jana\s+chaht[ai]"
    r"|jeena\s+nahi\s+chaht[ai]"
    r"|zinda\s+nahi\s+rehna\s+chaht[ai]"
    r"|jaan\s+de\s+d(?:e|u)n?"
    r"|apni\s+jaan\s+(?:le|de)\s*(?:l[ou]n|d[ou]n)?"
    r"|zindagi\s+khatam\s+kar"
    r"|apne\s+aap\s+ko\s+(?:khatam|nuqsan|zakhm[iy])"
    r"|khud\s+ko\s+(?:khatam|nuqsan)"
    r")\b",
    re.IGNORECASE,
)


def safety_hit(text: str) -> bool:
    """True if the text contains a self-harm phrase in English, Urdu
    script, or Roman Urdu. Deliberately over-inclusive."""
    if not text:
        return False
    s = str(text)
    if _EN_CRISIS.search(s):
        return True
    if _URDU_CRISIS.search(normalize_urdu(s)):
        return True
    if _ROMAN_URDU_CRISIS.search(s):
        return True
    return False


SAFETY_NOTICE = (
    "If you are thinking about harming yourself or ending your life, please "
    "reach out right now: contact your local emergency number or a crisis "
    "line (Pakistan: 0311-7786264 · Umang: 0317-4288665 · elsewhere: "
    "findahelpline.com). You are not alone, and talking to someone you "
    "trust can help."
)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Whisper ASR (Urdu)
# ═══════════════════════════════════════════════════════════════════════════

class WhisperEngine:
    def __init__(self, size: str = "tiny"):
        self.size = size
        self._lock = threading.Lock()
        self._tried = False
        self.backend: Optional[str] = None
        self.model = None
        self.error: Optional[str] = None

    def _load(self):
        if self._tried:
            return
        with self._lock:
            if self._tried:
                return
            self._tried = True
            errors = []

            # 1. Try faster-whisper (fastest on CPU)
            try:
                from faster_whisper import WhisperModel  # type: ignore
                self.model = WhisperModel(self.size, device="cpu", compute_type="int8", download_root="./hf_cache")
                self.backend = "faster-whisper"
                logger.info("Urdu ASR: faster-whisper-%s ready.", self.size)
                return
            except ImportError:
                errors.append("faster-whisper: not installed")
            except Exception as exc:
                errors.append(f"faster-whisper: {exc}")

            # 2. Try openai-whisper
            try:
                import whisper  # type: ignore
                self.model = whisper.load_model(self.size, download_root="./hf_cache")
                self.backend = "openai-whisper"
                logger.info("Urdu ASR: openai-whisper-%s ready.", self.size)
                return
            except ImportError:
                errors.append("openai-whisper: not installed")
            except Exception as exc:
                errors.append(f"openai-whisper: {exc}")

            self.error = " | ".join(errors)
            logger.warning("Urdu ASR unavailable: %s", self.error)

    def available(self) -> bool:
        self._load()
        return self.model is not None

    def transcribe(self, waveform: np.ndarray, sr: int = TARGET_SR) -> dict:
        base = {
            "available": False, "text": "", "segments": [],
            "language": "ur", "backend": None, "error": None,
        }
        if not self.available():
            base["error"] = self.error
            return base

        # Resample to 16 kHz mono float32 if needed (Whisper expects it)
        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != TARGET_SR:
            try:
                import librosa  # type: ignore
                wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
            except ImportError:
                # crude fallback: linear resample via numpy
                ratio = TARGET_SR / sr
                n_out = int(len(wav) * ratio)
                idx = np.linspace(0, len(wav) - 1, n_out)
                wav = np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)

        try:
            with self._lock:
                if self.backend == "faster-whisper":
                    segs_gen, _info = self.model.transcribe(
                        wav, language="ur", task="transcribe",
                        beam_size=5, vad_filter=True,
                        condition_on_previous_text=False,
                        no_speech_threshold=0.5,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                        initial_prompt="یہ ایک اردو گفتگو ہے۔",
                    )
                    segments = [
                        {"start": float(s.start), "end": float(s.end),
                         "text": s.text.strip(),
                         "no_speech_prob": float(getattr(s, "no_speech_prob", 0.0))}
                        for s in segs_gen
                    ]
                else:  # openai-whisper
                    res = self.model.transcribe(
                        wav, language="ur", task="transcribe",
                        beam_size=5, condition_on_previous_text=False,
                        fp16=False, verbose=None,
                    )
                    segments = [
                        {"start": float(s["start"]), "end": float(s["end"]),
                         "text": s["text"].strip(),
                         "no_speech_prob": float(s.get("no_speech_prob", 0.0))}
                        for s in res.get("segments", [])
                    ]
        except Exception as exc:
            base["error"] = f"transcription failed: {exc}"
            base["backend"] = self.backend
            return base

        # Normalize + filter hallucinations
        cleaned = []
        prev = None
        repeats = 0
        for s in segments:
            s["text"] = normalize_urdu(s["text"])
            if not s["text"]:
                continue
            if s.get("no_speech_prob", 0) > 0.6:
                continue
            repeats = repeats + 1 if s["text"] == prev else 0
            prev = s["text"]
            if repeats >= 2:
                continue
            cleaned.append(s)

        return {
            "available": True,
            "text": normalize_urdu(" ".join(s["text"] for s in cleaned)),
            "segments": cleaned,
            "language": "ur",
            "backend": self.backend,
            "model": self.size,
            "error": None,
        }


_asr = WhisperEngine()
def get_asr() -> WhisperEngine:
    return _asr


# ═══════════════════════════════════════════════════════════════════════════
# 4. Prosody extraction — librosa if available, numpy fallback otherwise
# ═══════════════════════════════════════════════════════════════════════════

PROSODY_KEYS = [
    "f0_mean", "f0_std_hz", "f0_std_st", "f0_range", "f0_slope_hz_s",
    "f0_voiced_frac", "f0_jitter",
    "rms_mean", "rms_std", "rms_dyn_db",
    "centroid_mean", "rolloff_mean", "flatness_mean", "zcr_mean",
    "band_low", "band_mid", "band_high",
    "pause_ratio", "pause_mean_sec",
    "speech_rate_wps",
] + [f"mfcc{i}_mean" for i in range(13)] + [f"mfcc{i}_std" for i in range(13)]

PROSODY_DIM = len(PROSODY_KEYS)


def _empty_prosody() -> Dict[str, float]:
    return {k: 0.0 for k in PROSODY_KEYS}


def _prosody_librosa(waveform: np.ndarray, sr: int,
                     word_count: Optional[float]) -> Dict[str, float]:
    import librosa

    f = _empty_prosody()
    y = np.asarray(waveform, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) < 2048:
        return f

    n_fft, hop = 1024, 256
    total = len(y) / sr
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))

    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    for i in range(13):
        f[f"mfcc{i}_mean"] = float(mfcc[i].mean())
        f[f"mfcc{i}_std"] = float(mfcc[i].std())

    # F0 via pyin (short clips) or yin (long)
    try:
        if total <= 90:
            f0, _, _ = librosa.pyin(y, fmin=65, fmax=500, sr=sr,
                                     frame_length=n_fft, hop_length=hop)
        else:
            f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr,
                              frame_length=n_fft, hop_length=hop)
            r = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
            med = float(np.median(r))
            f0 = np.where(r[:len(f0)] > 0.5 * med, f0, np.nan)

        valid = f0[np.isfinite(f0)]
        if valid.size >= 3:
            f["f0_mean"] = float(valid.mean())
            f["f0_std_hz"] = float(valid.std())
            f["f0_std_st"] = float(np.std(12 * np.log2(valid / valid.mean())))
            p5, p95 = np.percentile(valid, [5, 95])
            f["f0_range"] = float(p95 - p5)
            f["f0_voiced_frac"] = float(valid.size / len(f0))
            if valid.size >= 5:
                idx = np.where(np.isfinite(f0))[0]
                f["f0_slope_hz_s"] = float(np.polyfit(idx * hop / sr, valid, 1)[0])
                cons = np.diff(idx) == 1
                if cons.any():
                    f["f0_jitter"] = float(np.abs(np.diff(valid))[cons].mean() / valid.mean())
    except Exception as exc:
        logger.debug("F0 extraction skipped: %s", exc)

    rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)[0]
    f["rms_mean"] = float(rms.mean())
    f["rms_std"] = float(rms.std())
    db = 20 * np.log10(rms + 1e-8)
    f["rms_dyn_db"] = float(np.percentile(db, 95) - np.percentile(db, 5))

    f["centroid_mean"] = float(librosa.feature.spectral_centroid(S=S, sr=sr)[0].mean())
    f["rolloff_mean"] = float(librosa.feature.spectral_rolloff(S=S, sr=sr)[0].mean())
    f["flatness_mean"] = float(librosa.feature.spectral_flatness(S=S)[0].mean())
    f["zcr_mean"] = float(librosa.feature.zero_crossing_rate(
        y, frame_length=n_fft, hop_length=hop)[0].mean())

    P = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    tot = float(P.sum()) + 1e-12
    f["band_low"] = float(P[freqs < 500].sum() / tot)
    f["band_mid"] = float(P[(freqs >= 500) & (freqs < 2000)].sum() / tot)
    f["band_high"] = float(P[freqs >= 2000].sum() / tot)

    iv = librosa.effects.split(y, top_db=30, frame_length=n_fft, hop_length=hop)
    speech = float(sum(e - s for s, e in iv)) / sr
    gaps = [(iv[i + 1][0] - iv[i][1]) / sr for i in range(len(iv) - 1)]
    gaps = [g for g in gaps if g > 0.15]
    f["pause_ratio"] = float(max(0.0, 1.0 - speech / max(total, 1e-6)))
    f["pause_mean_sec"] = float(np.mean(gaps)) if gaps else 0.0
    if word_count:
        f["speech_rate_wps"] = float(word_count / total)
    return f


def _prosody_numpy(waveform: np.ndarray, sr: int,
                    word_count: Optional[float]) -> Dict[str, float]:
    """Minimal fallback using numpy only. Fewer features but always works."""
    f = _empty_prosody()
    y = np.asarray(waveform, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) < 1024:
        return f

    frame = int(sr * 0.025)
    hop = int(sr * 0.010)
    if len(y) < frame:
        return f
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = y[idx]

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    f["rms_mean"] = float(rms.mean())
    f["rms_std"] = float(rms.std())
    db = 20 * np.log10(rms + 1e-8)
    f["rms_dyn_db"] = float(np.percentile(db, 95) - np.percentile(db, 5))

    # crude autocorrelation pitch
    min_lag = max(1, int(sr / 500))
    max_lag = min(frame - 1, int(sr / 65))
    f0 = np.zeros(n, dtype=np.float32)
    voiced = np.zeros(n, dtype=bool)
    for i in range(n):
        seg = frames[i] - frames[i].mean()
        if np.sqrt(np.mean(seg ** 2)) < 1e-4:
            continue
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        if ac[0] <= 1e-8:
            continue
        ac = ac / ac[0]
        window = ac[min_lag:max_lag + 1]
        if window.size == 0:
            continue
        peak = int(np.argmax(window))
        if window[peak] < 0.3:
            continue
        f0[i] = sr / (peak + min_lag)
        voiced[i] = True

    valid = f0[voiced]
    if valid.size >= 3:
        f["f0_mean"] = float(valid.mean())
        f["f0_std_hz"] = float(valid.std())
        f["f0_std_st"] = float(np.std(12 * np.log2(valid / valid.mean())))
        f["f0_range"] = float(np.percentile(valid, 95) - np.percentile(valid, 5))
        f["f0_voiced_frac"] = float(valid.size / n)

    if word_count:
        f["speech_rate_wps"] = float(word_count / (len(y) / sr))
    return f


def extract_prosody(waveform: np.ndarray, sr: int,
                     word_count: Optional[float] = None
                     ) -> Tuple[np.ndarray, Dict[str, float], str]:
    """Returns (feature_vector, named_features_dict, engine_used).
    engine_used is 'librosa' or 'numpy'."""
    try:
        import librosa  # noqa: F401
        feats = _prosody_librosa(waveform, sr, word_count)
        engine = "librosa"
    except ImportError:
        feats = _prosody_numpy(waveform, sr, word_count)
        engine = "numpy"

    vec = np.array([feats.get(k, 0.0) for k in PROSODY_KEYS], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, feats, engine


# ═══════════════════════════════════════════════════════════════════════════
# 5. Convenience — enrich an SER result with Urdu + ASR + prosody + safety
# ═══════════════════════════════════════════════════════════════════════════

def enrich_audio_result(waveform: np.ndarray, sr: int,
                         existing: Optional[dict] = None) -> dict:
    """Return a dict of NEW fields to merge into the SER result.
    Never raises — any layer that fails contributes an 'error' string."""
    out: dict = {}

    # 3a. Transcribe
    try:
        asr = get_asr().transcribe(waveform, sr)
        out["transcript"] = asr
        text = asr.get("text", "")
    except Exception as exc:
        out["transcript"] = {"available": False, "text": "", "error": str(exc)}
        text = ""

    # 3b. Normalize + safety + sentiment analysis
    try:
        norm_text = normalize_urdu(text) if text else ""
        out["captured_text"] = text
        out["normalized_text"] = norm_text
        hit = safety_hit(text) if text else False
        out["safety_hit"] = hit
        out["safety_notice"] = SAFETY_NOTICE if hit else None

        # Run Urdu sentiment analysis on transcribed speech
        if text:
            try:
                from modules.text.urdu_sentiment import analyze_urdu_sentiment
                out["urdu_sentiment"] = analyze_urdu_sentiment(text)
            except Exception as e:
                out["urdu_sentiment"] = None
                out["urdu_sentiment_error"] = str(e)
        else:
            out["urdu_sentiment"] = None
    except Exception as exc:
        out["safety_hit"] = False
        out["safety_notice"] = None
        out["safety_error"] = str(exc)

    # 3c. Prosody
    try:
        word_count = len(text.split()) if text else None
        _, feats, engine = extract_prosody(waveform, sr, word_count=word_count)
        # Keep only finite floats, round for JSON
        out["prosody_features"] = {
            k: round(v, 5) for k, v in feats.items()
            if isinstance(v, (int, float)) and math.isfinite(v)
        }
        out["prosody_engine"] = engine
    except Exception as exc:
        out["prosody_features"] = {}
        out["prosody_engine"] = f"error: {exc}"

    return out


__all__ = [
    "normalize_urdu", "safety_hit", "SAFETY_NOTICE",
    "get_asr", "WhisperEngine",
    "extract_prosody", "PROSODY_KEYS", "PROSODY_DIM",
    "enrich_audio_result",
]