"""
modules/text/urdu_sentiment.py — Comprehensive Urdu Text Sentiment & Mental Health Analysis Engine.

Handles:
  1. Language identification (Urdu script, Roman Urdu, Mixed).
  2. Text canonicalization & normalization for Urdu script variants.
  3. Valence & Sentiment scoring (Positive, Neutral, Negative).
  4. Mental-health emotion classification (Happy, Sad, Angry, Anxious, Frustrated, Normal).
  5. Urdu negation handling (e.g. "خوش نہیں" -> sad, "سکون نہیں" -> anxious/sad).
  6. Crisis & self-harm detection with regional emergency hotlines.
  7. Clinical disorder correlation (Depression, Anxiety, Bipolar, PTSD).
  8. Culturally supportive therapeutic guidance & recommendations.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger("mindcare.text.urdu_sentiment")

# ── Character & Script Patterns ──────────────────────────────────────────────
_URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

_ROMAN_URDU_MARKERS = {
    "mein", "main", "mujhe", "mera", "meri", "mere", "hum", "humein", "hamara",
    "hai", "hain", "tha", "thi", "the", "nahi", "nahin", "nhi", "bohat", "boht",
    "bht", "bohot", "zyada", "zada", "kam", "kuch", "koi", "dil", "dimagh",
    "udaas", "udasi", "khush", "khushi", "pareshan", "pareshani", "gussa",
    "darr", "khauf", "dard", "takleef", "rona", "roti", "aansu", "sukoon",
    "khatam", "zindagi", "marna", "jeena", "theek", "acha", "bura", "yar",
}

_URDU_VARIANTS = {
    "ي": "ی", "ى": "ی",
    "ك": "ک",
    "ه": "ہ", "ۀ": "ہ", "ة": "ہ",
    "أ": "ا", "إ": "ا", "آ": "آ",
    "٠": "۰", "١": "۱", "٢": "۲", "٣": "۳", "٤": "۴",
    "٥": "۵", "٦": "۶", "٧": "۷", "٨": "۸", "٩": "۹",
}

_STRIP_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u0640\u200b-\u200f\ufeff]")

# ── Crisis / Emergency Phrases ───────────────────────────────────────────────
_URDU_CRISIS_RE = re.compile(
    r"(خود\s*کشی"
    r"|اپنی\s+جان\s+(?:لے|دے|دینا|لوں|دوں)"
    r"|مرنا\s+چاہت[ای]\s+ہوں"
    r"|مر\s*جانا\s+چاہت[ای]"
    r"|جینا\s+نہیں\s+چاہت[ای]"
    r"|زندہ\s+نہیں\s+رہنا\s+چاہت[ای]"
    r"|زندگی\s+ختم\s+کر"
    r"|اپنے\s+آپ\s+کو\s+(?:ختم|نقصان|زخمی)"
    r"|خود\s+کو\s+(?:ختم|نقصان|مار))"
)

_ROMAN_URDU_CRISIS_RE = re.compile(
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

URDU_SAFETY_NOTICE = (
    "اگر آپ خود کو نقصان پہنچانے یا زندگی ختم کرنے کے بارے میں سوچ رہے ہیں، "
    "تو براہ کرم فوری طور پر مدد حاصل کریں: پاکستان میں مفت ہیلپ لائن Umang پر رابطہ کریں "
    "(0317-4288665 / 0311-7786264) یا findahelpline.com پر رجوع کریں۔ آپ اکیلے نہیں ہیں، "
    "اور کسی بااعتماد انسان سے بات کرنا آپ کی مدد کر سکتا ہے۔"
)

# ── Emotion Lexicons for Urdu & Roman Urdu ────────────────────────────────────
URDU_LEXICON = {
    "happy": {
        "urdu": {
            "خوش", "خوشی", "خوشگوار", "مسکرا", "مسکراہٹ", "شکر", "الحمدللہ", "بہترین",
            "اچھا", "عمدہ", "شاندار", "سکون", "راحت", "امید", "محبت", "پیار", "کامیاب",
            "کامیابی", "اطمینان", "مبارک", "دلکش", "پرسکون", "خوشحال", "زندہ دل", "فخر",
            "لطف", "خوبصورت", "شادمانی", "تسلی", "جشن", "حوصلہ", "روشنی"
        },
        "roman": {
            "khush", "khushi", "sukoon", "behtareen", "acha", "achha", "shukar",
            "alhamdulillah", "umeed", "pyar", "kamiyab", "rahat", "mutmain", "enjoy",
            "hansi", "muskurahat", "proud", "shandar", "pyara", "khoobsurat"
        }
    },
    "sad": {
        "urdu": {
            "اداس", "اداسی", "غم", "غمگین", "رونا", "روتی", "روتا", "آنسو", "تنہا", "تنہائی",
            "اکیلا", "اکیلی", "مایوس", "مایوسی", "تکلیف", "درد", "مردہ", "بےکار", "بوجھ",
            "خالی", "دل ٹوٹ", "افسردہ", "مصیبت", "صدمہ", "حوصلہ ہار", "تباہ", "تاریکی",
            "اندھیرا", "محرومی", "برباد", "رنج", "پژمردہ", "بے بسی", "بیزاری", "موت",
            "اکتاہٹ", "نقصان", "حسرت", "بے رونق"
        },
        "roman": {
            "udaas", "udasi", "gham", "ghamgeen", "rona", "roti", "rota", "aansu",
            "tanha", "tanhai", "akela", "akeli", "mayoos", "mayoosi", "takleef", "dard",
            "bekaar", "bojh", "khali", "afsurda", "barbaad", "toot", "sad", "depression",
            "depressed", "bebas", "mehroom", "zulm", "bechain"
        }
    },
    "angry": {
        "urdu": {
            "غصہ", "غصے", "نفرت", "برا", "بدمزاج", "چیخ", "جھگڑا", "لڑائی", "مارنا", "تنگ",
            "بیزار", "پاگل", "خباثت", "سخت", "برہم", "اشتعال", "دشمنی", "ناراض", "ناراضگی",
            "بدلہ", "دھمکی", "ناراضگی", "حسد", "جلن", "کینہ", "بغض"
        },
        "roman": {
            "gussa", "gusse", "nafrat", "bura", "larai", "tang", "bizaar", "pagal",
            "naraaz", "narazgi", "cheekh", "dushmani", "badla", "irritate", "hate"
        }
    },
    "anxious": {
        "urdu": {
            "پریشان", "پریشانی", "خوف", "ڈر", "ڈرنا", "گھبراہٹ", "بے چینی", "بےچین",
            "دھڑکن", "فکر", "فکرمند", "سانس پھول", "پینک", "وسوسہ", "خطرہ", "سہم",
            "سہمنا", "دباؤ", "سٹریس", "ٹینشن", "ڈراونا", "ہراس", "اضطراب", "بے آرامی"
        },
        "roman": {
            "pareshan", "pareshani", "khauf", "darr", "ghabrahat", "bechaini", "fakar",
            "tension", "soch", "overthinking", "stress", "panic", "dar", "khof", "dhadkan"
        }
    },
    "frustrated": {
        "urdu": {
            "الجھن", "تھک گیا", "تھک گئی", "اکتاہٹ", "بےبس", "عاجز", "بیوس", "پھنس گیا",
            "بیزار", "بے بس", "ناکام", "رکاوٹ", "مسئلہ"
        },
        "roman": {
            "uljhan", "thak gaya", "thak gayi", "bebas", "aajiz", "frustrate", "fas gaya",
            "tang agaya", "tang a gayi", "thak"
        }
    },
    "normal": {
        "urdu": {
            "ٹھیک", "عام", "روزمرہ", "معمول", "صحیح", "کام", "آج", "بات", "خبر",
            "کتاب", "گھر", "دفتر", "پڑھائی", "دوست", "طبیعت", "حالات"
        },
        "roman": {
            "theek", "aam", "normal", "kam", "baat", "routine", "rozmarrah", "sahi", "chal"
        }
    }
}

URDU_NEGATORS = {
    "نہیں", "نہ", "مت", "کبھی نہیں", "کوئی نہیں", "بغیر", "بنا", "نقصان نہیں",
    "nahi", "nahin", "nhi", "mat", "kabhi nahi", "bina", "baghair", "no", "not"
}

URDU_INTENSIFIERS = {
    "بہت", "بے حد", "انتہائی", "شدید", "سخت", "زیادہ", "حد سے زیادہ", "مکمل",
    "bohat", "boht", "bht", "bohot", "zyada", "intehai", "shadeed", "sakht", "puree"
}


def is_urdu_text(text: str) -> bool:
    """True if text contains Urdu Arabic-script characters or distinct Roman Urdu words."""
    if not text or not isinstance(text, str):
        return False
    # Check script
    if _URDU_SCRIPT_RE.search(text):
        return True
    # Check Roman Urdu keywords
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return False
    urdu_word_count = sum(1 for w in words if w in _ROMAN_URDU_MARKERS)
    return urdu_word_count >= 2 or (len(words) <= 3 and urdu_word_count >= 1)


def normalize_urdu_script(text: str) -> str:
    """Canonicalize Urdu script variants for robust lexical matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(_URDU_VARIANTS.get(c, c) for c in text)
    text = _STRIP_DIACRITICS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def check_urdu_safety(text: str) -> Tuple[bool, Optional[str]]:
    """Checks for self-harm or emergency crisis phrases in Urdu and Roman Urdu."""
    if not text:
        return False, None
    s = str(text)
    norm = normalize_urdu_script(s)
    if _URDU_CRISIS_RE.search(norm) or _ROMAN_URDU_CRISIS_RE.search(s):
        return True, URDU_SAFETY_NOTICE
    return False, None


def analyze_urdu_sentiment(text: str) -> Dict[str, Any]:
    """
    Performs sentiment analysis and mental-health emotion categorization
    on Urdu (Arabic script or Roman transliteration).

    Returns the contract matching format_result() in app.py:
      {
        "input_text": text,
        "clean_text": normalized_text,
        "selected_emotion": "sad",
        "emotion": "sad",
        "emotion_confidence": 0.88,
        "confidence": 0.88,
        "sentiment": "negative",
        "sentiment_score": -0.75,
        "emotion_probabilities": {...},
        "disorder_probabilities": {...},
        "top_disorder": {...},
        "urgent": bool,
        "safety_notice": Optional[str],
        "language": "ur" | "ur-Latn",
        "recommendations": {...}
      }
    """
    raw = str(text).strip()
    is_script = bool(_URDU_SCRIPT_RE.search(raw))
    language_code = "ur" if is_script else "ur-Latn"
    normalized = normalize_urdu_script(raw) if is_script else raw.lower()

    # Safety check
    is_urgent, safety_notice = check_urdu_safety(raw)

    # Tokenize
    if is_script:
        tokens = [t.strip("،۔؟!؛:") for t in normalized.split() if t.strip("،۔؟!؛:")]
    else:
        tokens = re.findall(r"\b[a-zA-Z']+\b", normalized.lower())

    # Score emotions with negation & intensifier context
    scores: Counter = Counter()
    for emo in URDU_LEXICON:
        scores[emo] = 0.1  # baseline smoothing

    total_tokens = max(1, len(tokens))
    for i, tok in enumerate(tokens):
        # Look around for negators and intensifiers
        lookback = set(tokens[max(0, i - 3):i])
        lookahead = set(tokens[i + 1:min(len(tokens), i + 3)])
        has_negator = bool(URDU_NEGATORS & (lookback | lookahead))
        has_intensifier = bool(URDU_INTENSIFIERS & lookback)
        weight = 2.0 if has_intensifier else 1.2

        for emo, lex_dict in URDU_LEXICON.items():
            vocab = lex_dict["urdu"] if is_script else lex_dict["roman"]
            if tok in vocab:
                if has_negator:
                    if emo == "happy":
                        scores["sad"] += weight * 1.5
                        scores["anxious"] += weight * 0.8
                    elif emo in ("sad", "angry", "anxious"):
                        scores["normal"] += weight * 1.2
                        scores[emo] = max(0.0, scores[emo] - 0.5)
                else:
                    scores[emo] += weight

    # If safety triggered, heavily elevate sad/depression
    if is_urgent:
        scores["sad"] += 5.0
        scores["anxious"] += 3.0

    # Probability normalization
    all_emotions = ["normal", "happy", "sad", "angry", "frustrated", "anxious"]
    raw_vals = np.array([scores[e] for e in all_emotions], dtype=np.float64)
    exp_vals = np.exp(raw_vals - np.max(raw_vals))
    probs = exp_vals / np.sum(exp_vals)
    prob_dict = {e: round(float(p), 4) for e, p in zip(all_emotions, probs)}

    selected_emotion = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[selected_emotion]

    # Map sentiment polarity
    pos_score = prob_dict["happy"]
    neg_score = prob_dict["sad"] + prob_dict["angry"] + prob_dict["anxious"] + prob_dict["frustrated"]
    if selected_emotion == "normal" and not is_urgent and neg_score < 0.6:
        sentiment = "neutral"
        sentiment_score = 0.0
    elif pos_score > neg_score + 0.10:
        sentiment = "positive"
        sentiment_score = round(pos_score - neg_score, 3)
    elif neg_score > pos_score + 0.10:
        sentiment = "negative"
        sentiment_score = round(pos_score - neg_score, 3)
    else:
        sentiment = "neutral"
        sentiment_score = 0.0

    # Map clinical mental health disorder correlations
    # Trained disorders: adhd, aspergers, depression, ocd, ptsd
    # Untrained disorders: anxiety_disorder, bipolar, bpd, schizophrenia
    dep_score = min(0.98, prob_dict["sad"] * 1.3 + (0.3 if is_urgent else 0.0))
    anx_score = min(0.95, prob_dict["anxious"] * 1.2 + prob_dict["frustrated"] * 0.3)
    ptsd_score = min(0.92, (prob_dict["anxious"] + prob_dict["sad"]) * 0.55 + (0.4 if is_urgent else 0.0))
    bipolar_score = min(0.85, (prob_dict["happy"] * prob_dict["angry"]) * 2.0 + prob_dict["frustrated"] * 0.4)

    disorder_probabilities = {
        "adhd": round(min(0.85, prob_dict["frustrated"] * 0.6 + 0.05), 3),
        "anxiety_disorder": round(anx_score, 3),
        "aspergers": 0.04,
        "bipolar": round(bipolar_score, 3),
        "bpd": round(min(0.90, (prob_dict["sad"] + prob_dict["angry"]) * 0.45), 3),
        "depression": round(dep_score, 3),
        "ocd": round(min(0.80, prob_dict["anxious"] * 0.5 + 0.05), 3),
        "ptsd": round(ptsd_score, 3),
        "schizophrenia": round(min(0.70, (prob_dict["anxious"] + (0.3 if is_urgent else 0.0)) * 0.4), 3),
    }

    top_name = max(disorder_probabilities, key=disorder_probabilities.get)
    top_prob = disorder_probabilities[top_name]
    top_disorder = (
        {"name": top_name, "probability": top_prob}
        if top_prob >= 0.50
        else None
    )

    # Urdu-friendly empathetic recommendations
    if is_urgent:
        recs = {
            "title": "فوری مدد اور معاونت (Urgent Support)",
            "urgent": True,
            "consolation": "آپ کا تحفظ اور زندگی بے حد قیمتی ہے۔ اس مشکل گھڑی میں اکیلے نہ رہیں، ہمدرد افراد اور ماہرین آپ کی مدد کے لیے موجود ہیں۔",
            "steps": [
                "پاکستان میں مفت مینٹل ہیلتھ ہیلپ لائن Umang پر کال کریں: 0317-4288665",
                "اپنے کسی بااعتماد دوست، بہن بھائی یا والدین سے فوری بات کریں۔",
                "قریبی ہسپتال یا کلینک کے شعبہ ایمرجنسی سے رابطہ کریں۔",
                "اگر آپ بیرون ملک ہیں تو findahelpline.com پر اپنے ملک کی فری کرائسز سروس منتخب کریں۔"
            ]
        }
    elif selected_emotion in ("sad", "anxious", "frustrated"):
        recs = {
            "title": "ذہنی سکون اور بحالی کے اقدامات (Self-Care & Calm)",
            "urgent": False,
            "consolation": "ذہنی تناؤ یا اداسی کا احساس وقتی ہو سکتا ہے۔ اپنے آپ پر سختی نہ کریں اور آرام کو ترجیح دیں۔",
            "steps": [
                "گہری سانس لینے کی مشق (Deep Breathing): 4 سیکنڈ اندر، 4 سیکنڈ روکیں، اور 4 سیکنڈ باہر نکالیں۔",
                "کسی اچھے دوست یا قریبی ساتھی سے دل کی بات شیئر کریں۔",
                "روزانہ 20 منٹ کھلی ہوا میں چہل قدمی کریں اور نیند کا خیال رکھیں۔",
                "اگر اداسی طویل عرصے سے برقرار ہے تو مستند کونسلر یا سائیکالوجسٹ سے مشورہ لیں۔"
            ]
        }
    elif selected_emotion == "angry":
        recs = {
            "title": "غصے پر قابو اور جذباتی توازن (Emotional Regulation)",
            "urgent": False,
            "consolation": "غصہ ایک فطری جذبہ ہے، مگر اس کا مثبت طریقے سے اظہار آپ کی ذہنی صحت کو بہتر بناتا ہے۔",
            "steps": [
                "فوری ردعمل دینے سے گریز کریں اور 10 تک گنتی گن کر گہرا سانس لیں۔",
                "ٹھنڈا پانی پیئیں اور تھوڑی دیر کے لیے ماحول کو تبدیل کریں۔",
                "اپنے جذبات کو ڈائری میں لکھ کر باہر نکالیں۔"
            ]
        }
    else:
        recs = {
            "title": "مثبت توانائی اور توازن (Healthy Mindset)",
            "urgent": False,
            "consolation": "ماشاءاللہ آپ کا ذہنی مزاج مثبت اور متوازن نظر آ رہا ہے۔ اس کیفیت کو برقرار رکھیں۔",
            "steps": [
                "اپنی صحت بخش عادات، شکر گزاری اور معمولات کو جاری رکھیں۔",
                "دوسروں کے ساتھ خوشیاں بانٹیں اور ان کی ہمت بندھائیں۔"
            ]
        }

    # Style mapping
    emotion_styles = {
        "happy": {"icon": "😊", "color": "#22C55E"},
        "sad": {"icon": "😢", "color": "#3B82F6"},
        "angry": {"icon": "😠", "color": "#EF4444"},
        "anxious": {"icon": "😰", "color": "#A78BFA"},
        "frustrated": {"icon": "😤", "color": "#F97316"},
        "normal": {"icon": "😐", "color": "#64748B"},
    }
    style = emotion_styles.get(selected_emotion, {"icon": "🙂", "color": "#94A3B8"})

    return {
        "input_text": raw,
        "clean_text": normalized,
        "clean_tokens": tokens,
        "token_ids": [1] * min(len(tokens), 50),
        "padded_length": 50,
        "truncated": False,
        "language": language_code,
        "language_name": "Urdu" if is_script else "Roman Urdu",
        "selected_emotion": selected_emotion,
        "emotion": selected_emotion,
        "emotion_confidence": confidence,
        "confidence": confidence,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "icon": style["icon"],
        "color": style["color"],
        "emotion_probabilities": prob_dict,
        "disorder_probabilities": disorder_probabilities,
        "trained_disorders": ["adhd", "aspergers", "depression", "ocd", "ptsd"],
        "untrained_disorders": ["anxiety_disorder", "bipolar", "bpd", "schizophrenia"],
        "top_disorder": top_disorder,
        "urgent": is_urgent,
        "safety_notice": safety_notice,
        "recommendations": recs,
        "model_used": "Urdu NLP Multilingual Emotion Engine",
        "is_urdu": True,
    }
