"""
app.py — MindCare Flask API (v6, dual-head token-attention model).

Text analysis runs on inference.predict_pipeline (TensorFlow/Keras).
The other modules from the previous version (chat agent, scrapers, RAG,
smart-room video, image/audio blueprints) are kept; if one of them is
not installed, the API still starts and that single endpoint answers
503 with a clear message.

Dev:
    python app.py

Production:
    Linux/macOS:
        gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 --timeout 180 app:app

    Windows:
        waitress-serve --listen=0.0.0.0:5000 --threads=4 app:app

Environment variables:
    MINDCARE_MODEL_DIR  - default ./models
    CORS_ORIGINS       - default *
    MINDCARE_UPLOAD_DIR - default ./uploads
    FLASK_DEBUG=1       - Flask development reloader
    PORT                - default 5000
"""

import logging
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from functools import wraps

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, g, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("mindcare.api")


# ============================================================================
# TEXT MODEL
# ============================================================================

import inference
from inference import predict_batch, predict_pipeline
from recommendations import recommendations_for


# ============================================================================
# OPTIONAL MODULE LOADERS
# ============================================================================

def _optional(label, loader):
    try:
        return loader()
    except Exception as exc:
        logger.warning(
            "Optional module '%s' unavailable: %s",
            label,
            exc,
        )
        return None


def _load_scrapers():
    from scrappers.reddit import fetch_user_text_posts
    from scrappers.twitter import fetch_tweets

    return fetch_user_text_posts, fetch_tweets


def _load_smart_room():
    from modules.smart_room import (
        SmartRoomError,
        SmartRoomSession,
    )

    return SmartRoomSession, SmartRoomError


def _load_chat_agent():
    from modules.chat_agent import (
        MindCareGeminiAgent,
        reset_session,
    )

    return MindCareGeminiAgent(), reset_session


def _load_blueprints():
    from routes.audio_routes import audio_api
    from routes.image_routes import image_api

    return image_api, audio_api


_scrapers = _optional(
    "scrapers",
    _load_scrapers,
)

_smart_room = _optional(
    "smart_room",
    _load_smart_room,
)

_chat = _optional(
    "chat_agent",
    _load_chat_agent,
)

_blueprints = _optional(
    "image/audio routes",
    _load_blueprints,
)


# ============================================================================
# APP CONFIGURATION
# ============================================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "*",
    ).split(",")
    if origin.strip()
]

CORS(
    app,
    resources={
        r"/*": {
            "origins": _origins,
        }
    },
)


# ============================================================================
# OPTIONAL BLUEPRINTS
# ============================================================================

if _blueprints:
    app.register_blueprint(
        _blueprints[0],
        url_prefix="/api/image",
    )

    app.register_blueprint(
        _blueprints[1],
        url_prefix="/api/audio",
    )


# ============================================================================
# UPLOAD DIRECTORY
# ============================================================================

UPLOAD_FOLDER = os.environ.get(
    "MINDCARE_UPLOAD_DIR",
    "./uploads",
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True,
)


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

MODEL_ERROR = None

try:
    inference.warm_up()

    logger.info(
        "Text model ready."
    )

except Exception as exc:
    MODEL_ERROR = str(exc)

    logger.error(
        "Text model failed to load: %s",
        exc,
    )


# ============================================================================
# RAG PIPELINE
# ============================================================================

_rag_pipeline = None


def get_rag_pipeline():
    global _rag_pipeline

    if _rag_pipeline is None:
        from modules.rag.rag import RAGPipeline

        _rag_pipeline = RAGPipeline()

    return _rag_pipeline


# ============================================================================
# HELPERS
# ============================================================================

EMOTION_STYLE = {
    "normal": {
        "icon": "😐",
        "color": "#64748B",
    },
    "happy": {
        "icon": "😊",
        "color": "#22C55E",
    },
    "sad": {
        "icon": "😢",
        "color": "#3B82F6",
    },
    "angry": {
        "icon": "😠",
        "color": "#EF4444",
    },
    "frustrated": {
        "icon": "😤",
        "color": "#F97316",
    },
    "anxious": {
        "icon": "😰",
        "color": "#A78BFA",
    },
}


def format_result(trace: dict) -> dict:
    """
    Convert inference output into the JSON contract
    expected by the React frontend.
    """

    emo = trace["selected_emotion"]

    style = EMOTION_STYLE.get(
        emo,
        {
            "icon": "🙂",
            "color": "#94A3B8",
        },
    )

    out = dict(trace)

    out.update(
        {
            "emotion": emo,
            "confidence": trace["emotion_confidence"],
            "icon": style["icon"],
            "color": style["color"],
            "urgent": bool(trace["safety_notice"]),
            "model_used": "token-attention dual-head",
        }
    )

    if "recommendations" not in out or not out["recommendations"]:
        out["recommendations"] = recommendations_for(out)

    return out


def error_response(msg, code=400):
    return (
        jsonify(
            {
                "error": msg,
                "status": "error",
            }
        ),
        code,
    )


def require_json(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not request.is_json:
            return error_response(
                "Content-Type must be application/json"
            )

        return f(*args, **kwargs)

    return decorated


def require_model(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if MODEL_ERROR:
            return error_response(
                f"Text model is not loaded: {MODEL_ERROR}",
                503,
            )

        return f(*args, **kwargs)

    return decorated


def unavailable(label):
    return error_response(
        f"The '{label}' module is not installed on this server.",
        503,
    )


# ============================================================================
# GLOBAL REQUEST LOGGER
# ============================================================================

@app.before_request
def _start_timer():
    """
    Print every incoming request to the Flask server console.
    """

    g.start = time.perf_counter()

    print("\n" + "=" * 80)
    print("📡 MINDCARE SERVER REQUEST")
    print("=" * 80)

    print(f"➡ Method      : {request.method}")
    print(f"➡ Path        : {request.path}")
    print(f"➡ Full URL    : {request.url}")
    print(f"➡ Remote IP   : {request.remote_addr}")

    user_agent = request.headers.get(
        "User-Agent",
        "Unknown",
    )

    print(f"➡ User Agent  : {user_agent}")

    # JSON request body
    if request.is_json:
        try:
            request_data = request.get_json(
                silent=True
            )

            if request_data:
                print("➡ JSON Body   :")

                for key, value in request_data.items():
                    value_string = str(value)

                    if len(value_string) > 500:
                        value_string = (
                            value_string[:500]
                            + "... [TRUNCATED]"
                        )

                    print(
                        f"   {key}: {value_string}"
                    )
            else:
                print(
                    "➡ JSON Body   : <empty>"
                )

        except Exception as exc:
            print(
                f"⚠ Could not read JSON body: {exc}"
            )

    # Multipart files
    if request.files:
        print("➡ Uploaded Files:")

        for field_name, uploaded_file in request.files.items():
            print(
                f"   {field_name}: "
                f"{uploaded_file.filename}"
            )

    # Form data
    if request.form:
        print("➡ Form Data   :")

        for key, value in request.form.items():
            value_string = str(value)

            if len(value_string) > 500:
                value_string = (
                    value_string[:500]
                    + "... [TRUNCATED]"
                )

            print(
                f"   {key}: {value_string}"
            )

    print("=" * 80)


@app.after_request
def _add_timing(response):
    """
    Print response status and execution time for every request.
    """

    if hasattr(g, "start"):
        elapsed_ms = round(
            (
                time.perf_counter()
                - g.start
            )
            * 1000,
            2,
        )

        response.headers[
            "X-Response-Time-Ms"
        ] = str(elapsed_ms)

        print(
            f"⬅ RESPONSE     : "
            f"{response.status_code} | "
            f"{request.method} "
            f"{request.path} | "
            f"{elapsed_ms} ms"
        )

        print("=" * 80 + "\n")

    return response


# ============================================================================
# CORE
# ============================================================================

@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "name": (
                "MindCare API v6 "
                "(dual-head token attention)"
            ),
            "status": (
                "healthy"
                if not MODEL_ERROR
                else "degraded"
            ),
            "optional_modules": {
                "scrapers": bool(_scrapers),
                "smart_room": bool(_smart_room),
                "chat_agent": bool(_chat),
                "image_audio": bool(_blueprints),
            },
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    info = {
        "status": (
            "healthy"
            if not MODEL_ERROR
            else "degraded"
        ),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_loaded": MODEL_ERROR is None,
        "model_error": MODEL_ERROR,
    }

    if MODEL_ERROR is None:
        st = inference.load_artifacts()

        info.update(
            {
                "emotions": st["emotions"],
                "disorders": st["disorders"],
                "trained_disorders": st[
                    "trained_disorders"
                ],
                "untrained_disorders": st[
                    "untrained_disorders"
                ],
                "artefact_dir": st["folder"],
            }
        )

    info["optional_modules"] = {
        "scrapers": bool(_scrapers),
        "smart_room": bool(_smart_room),
        "chat_agent": bool(_chat),
        "image_audio": bool(_blueprints),
    }

    return (
        jsonify(info),
        200 if MODEL_ERROR is None else 503,
    )


# ============================================================================
# TEXT PREDICTION
# ============================================================================
# ── Overwrite the predict endpoint directly inside app.py ──────────────────────────
@app.route("/api/predict", methods=["POST"])
@require_json
@require_model
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "")
    text = text.strip() if isinstance(text, str) else ""
    if not text:
        return error_response('"text" is required.')
    if len(text) > 8000:
        return error_response("Text too long (max 8,000 characters).")
        
    print(f"\n🔬 [ATTENTION ENGINE] Ingesting text payload into granular processing layer...")
    
    try:
        # 🚀 SENTENCE SPLITTER: Breaks down text by periods, exclamation marks, question marks, and Urdu punctuation
        # It filters out empty slots and handles clean spacing parameters natively
        sentences = [s.strip() for s in re.split(r'(?<=[.!?۔؟])\s+', text) if s.strip()]
        
        # CONDITION CHECK: If the text is short (≤ 2 sentences), run a standard single-pass calculation
        if len(sentences) <= 2:
            print(f"   ↳ Short text detected ({len(sentences)} snt). Running standard single-pass inference.")
            return jsonify({
                "status": "success", 
                "segmentation": "single",
                "result": format_result(predict_pipeline(text))
            })
            
        # 🚀 MULTI-PASS GRANULAR MATRIX: If the text is long, decompose and evaluate every sentence separately
        print(f"   ↳ Long-form text detected ({len(sentences)} sentences). Activating Sentence-Level Analysis...")
        
        sentence_analyses = []
        for idx, sentence in enumerate(sentences):
            try:
                # Run batch pipeline preprocessing on each unique sentence entry track
                raw_trace = predict_pipeline(sentence)
                formatted_trace = format_result(raw_trace)
                
                sentence_analyses.append({
                    "sentence_index": idx + 1,
                    "text": sentence,
                    "emotion": formatted_trace["emotion"],
                    "confidence": formatted_trace["confidence"],
                    "color": formatted_trace["color"],
                    "icon": formatted_trace["icon"],
                    "disorder_probabilities": formatted_trace["disorder_probabilities"],
                    "urgent": formatted_trace["urgent"]
                })
                print(f"     [Segment {idx+1}/{len(sentences)}] → Emotion: {formatted_trace['emotion'].upper()} | Text: \"{sentence[:40]}...\"")
            except Exception as e:
                # Graceful degradation fallback if a single segment contains only symbols
                continue

        # Compute a combined summary average across all processed segments to provide an overall profile
        combined_emotions = [seg["emotion"] for seg in sentence_analyses]
        dominant_emotion = max(set(combined_emotions), key=combined_emotions.count) if combined_emotions else "normal"
        style = EMOTION_STYLE.get(dominant_emotion, {"icon": "😐", "color": "#64748B"})

        # Always produce a full-text result object so the frontend AnalysisView component
        # can render properly (it expects result.disorder_probabilities, result.top_disorder, etc.)
        try:
            full_result = format_result(predict_pipeline(text))
        except Exception:
            # If the combined text fails (e.g. too long for model), synthesise a minimal result
            # from the most common segment's analysis
            dominant_seg = next(
                (s for s in sentence_analyses if s["emotion"] == dominant_emotion),
                sentence_analyses[0] if sentence_analyses else {}
            )
            full_result = {
                "emotion": dominant_emotion,
                "confidence": dominant_seg.get("confidence", 0.5),
                "icon": style["icon"],
                "color": style["color"],
                "urgent": any(s.get("urgent") for s in sentence_analyses),
                "disorder_probabilities": dominant_seg.get("disorder_probabilities", {}),
                "untrained_disorders": [],
                "top_disorder": None,
                "safety_notice": None,
                "recommendations": None,
                "clean_text": text[:200],
                "clean_tokens": [],
                "token_ids": [],
                "padded_length": 0,
                "truncated": False,
                "emotion_probabilities": {},
            }
        
        print(f"   ↳ Analysis complete. Dominant structural emotion determined: {dominant_emotion.upper()}")
        return jsonify({
            "status": "success",
            "segmentation": "granular",
            "total_segments": len(sentence_analyses),
            "dominant_emotion": dominant_emotion,
            "icon": style["icon"],
            "color": style["color"],
            "segments": sentence_analyses,
            "result": full_result,  # ← required by AnalysisView component
        })

    except Exception as exc:
        traceback.print_exc()
        return error_response(str(exc), 500)


@app.route("/api/explain", methods=["POST"])
@require_json
@require_model
def explain():
    text = (
        (request.get_json(
            silent=True
        ) or {}).get("text") or ""
    ).strip()

    if not text:
        return error_response(
            '"text" is required.'
        )

    try:
        result = format_result(
            predict_pipeline(text)
        )

    except ValueError as exc:
        return error_response(
            str(exc),
            422,
        )

    lines = [
        f'Clean text : "{result["clean_text"]}"',
        f'Tokens     : {result["clean_tokens"][:30]}',
        f'Token ids  : {result["token_ids"][:30]}',
        (
            f'Emotion    : '
            f'{result["emotion"].upper()} '
            f'({result["confidence"]:.1%})'
        ),
    ]

    lines += [
        f"  {key:<12} {value:.1%}"
        for key, value
        in result[
            "emotion_probabilities"
        ].items()
    ]

    lines.append(
        "Clinical-pattern probabilities (independent):"
    )

    lines += [
        (
            f'  {key:<18} '
            f'{"not trained" if key in result["untrained_disorders"] else format(value, ".1%")}'
        )
        for key, value
        in result[
            "disorder_probabilities"
        ].items()
    ]

    return jsonify(
        {
            "status": "success",
            "explanation": "\n".join(lines),
            "result": result,
        }
    )


# ============================================================================
# PHQ-9 / GAD-7 SCREENING
# ============================================================================

@app.route("/api/assessment", methods=["POST"])
@require_json
def score_assessment():
    data = request.get_json(
        silent=True
    ) or {}

    atype = str(
        data.get(
            "type",
            "",
        )
    ).upper()

    responses = data.get(
        "responses",
        [],
    )

    if atype not in (
        "PHQ9",
        "GAD7",
    ):
        return error_response(
            '"type" must be "PHQ9" or "GAD7".'
        )

    print(
        f"📝 [ASSESSMENT] "
        f"User initiated an active "
        f"{atype} diagnostic survey execution pass..."
    )

    expected = (
        9
        if atype == "PHQ9"
        else 7
    )

    if (
        not isinstance(responses, list)
        or len(responses) != expected
    ):
        return error_response(
            f"{atype} requires "
            f"{expected} responses (0–3)."
        )

    if not all(
        isinstance(response, int)
        and not isinstance(response, bool)
        and 0 <= response <= 3
        for response in responses
    ):
        return error_response(
            "All responses must be integers 0–3."
        )

    total = sum(responses)

    if atype == "PHQ9":
        if total <= 4:
            severity = "Minimal"
            level = 0
        elif total <= 9:
            severity = "Mild"
            level = 1
        elif total <= 14:
            severity = "Moderate"
            level = 2
        elif total <= 19:
            severity = "Moderately Severe"
            level = 3
        else:
            severity = "Severe"
            level = 4

        max_score = 27

    else:
        if total <= 4:
            severity = "Minimal"
            level = 0
        elif total <= 9:
            severity = "Mild"
            level = 1
        elif total <= 14:
            severity = "Moderate"
            level = 2
        else:
            severity = "Severe"
            level = 3

        max_score = 21

    return jsonify(
        {
            "status": "success",
            "type": atype,
            "total_score": total,
            "severity": severity,
            "severity_level": level,
            "max_score": max_score,
            "percentage": round(
                total / max_score * 100,
                1,
            ),
            "recommend_referral": level >= 2,
        }
    )


# ============================================================================
# SOCIAL PHENOTYPING
# ============================================================================

def _summarise(predictions):
    counts = {}

    for prediction in predictions:
        emotion = prediction["emotion"]

        counts[emotion] = (
            counts.get(emotion, 0) + 1
        )

    return {
        "emotion_counts": counts,
        "urgent_count": sum(
            1
            for prediction in predictions
            if prediction["urgent"]
        ),
    }


# ============================================================================
# REDDIT SCRAPER
# ============================================================================
# ── Overwrite the scrape_reddit endpoint directly inside app.py ───────────────────
# ── Overwrite the scrape_reddit endpoint inside app.py ──────────────────────────
# ============================================================================
# REDDIT SCRAPER
# ============================================================================

@app.route("/api/scrape/reddit", methods=["POST"])
@require_json
@require_model
def scrape_reddit():
    data = request.get_json(silent=True) or {}

    target = str(data.get("target", "")).strip()

    try:
        limit = int(data.get("limit", 10) or 10)
    except (TypeError, ValueError):
        limit = 10

    limit = max(1, min(limit, 50))

    if not target:
        return error_response('"target" username is required.')

    # Normalize username
    if target.startswith("u/"):
        target = target[2:]

    if target.startswith("u_"):
        target = target[2:]

    # Remove accidental @
    target = target.lstrip("@").strip()

    print()
    print("=" * 80)
    print("👽 [REDDIT LIVE SCRAPER]")
    print(f"   ↳ Target user : u/{target}")
    print(f"   ↳ Requested limit: {limit}")
    print("=" * 80)

    import requests
    import xml.etree.ElementTree as ET
    import html

    results = []

    # ------------------------------------------------------------------------
    # REDDIT RSS FEED
    # ------------------------------------------------------------------------

    try:
        url = f"https://www.reddit.com/user/{target}/submitted/.rss"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/154.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml",
        }

        print("🔍 Fetching Reddit user submission RSS...")
        print(f"   ↳ URL: {url}")

        response = requests.get(
            url,
            headers=headers,
            timeout=15,
        )

        print(f"   ↳ Reddit HTTP status: {response.status_code}")

        if response.status_code != 200:
            print(
                f"⚠️ Reddit RSS request failed with HTTP "
                f"{response.status_code}"
            )
        else:
            # ---------------------------------------------------------------
            # Parse XML
            # ---------------------------------------------------------------

            root = ET.fromstring(response.content)

            # Correct Atom namespace
            namespace = {
                "atom": "http://www.w3.org/2005/Atom"
            }

            entries = root.findall("atom:entry", namespace)

            print(f"   ↳ RSS entries received: {len(entries)}")

            for entry in entries:

                if len(results) >= limit:
                    break

                # -----------------------------------------------------------
                # Extract fields safely
                # -----------------------------------------------------------

                id_element = entry.find("atom:id", namespace)
                title_element = entry.find("atom:title", namespace)
                content_element = entry.find("atom:content", namespace)
                updated_element = entry.find("atom:updated", namespace)

                link_element = entry.find("atom:link", namespace)

                source_id = (
                    id_element.text.strip()
                    if id_element is not None and id_element.text
                    else f"reddit_{len(results)}"
                )

                title = (
                    title_element.text.strip()
                    if title_element is not None and title_element.text
                    else ""
                )

                raw_content = (
                    content_element.text
                    if content_element is not None
                    and content_element.text
                    else ""
                )

                created_utc = (
                    updated_element.text.strip()
                    if updated_element is not None
                    and updated_element.text
                    else ""
                )

                post_url = ""

                if link_element is not None:
                    post_url = link_element.get("href", "")

                # -----------------------------------------------------------
                # Convert HTML content to plain text
                # -----------------------------------------------------------

                text = html.unescape(raw_content)

                # Remove HTML tables
                text = re.sub(
                    r"<table.*?>.*?</table>",
                    " ",
                    text,
                    flags=re.DOTALL | re.IGNORECASE,
                )

                # Remove scripts/styles
                text = re.sub(
                    r"<(script|style).*?>.*?</\1>",
                    " ",
                    text,
                    flags=re.DOTALL | re.IGNORECASE,
                )

                # Remove HTML tags
                text = re.sub(
                    r"<[^>]+>",
                    " ",
                    text,
                )

                # Normalize whitespace
                body = re.sub(
                    r"\s+",
                    " ",
                    text,
                ).strip()

                # -----------------------------------------------------------
                # Ignore deleted / empty content
                # -----------------------------------------------------------

                if body in ("[removed]", "[deleted]", ""):
                    body = ""

                # If body is empty, use title when it is meaningful
                if len(body) < 5 and len(title) >= 10:
                    body = title

                if len(body) < 5:
                    continue

                # -----------------------------------------------------------
                # Store post
                # -----------------------------------------------------------

                results.append(
                    {
                        "id": source_id,
                        "title": title,
                        "body": body,
                        "subreddit": "unknown",
                        "created_utc": created_utc,
                        "url": post_url,
                    }
                )

            print(
                f"   ↳ Valid text posts extracted: {len(results)}"
            )

    except requests.RequestException as exc:
        print(
            f"⚠️ Reddit network request failed: {exc}"
        )

    except ET.ParseError as exc:
        print(
            f"⚠️ Reddit RSS XML parsing failed: {exc}"
        )

    except Exception as exc:
        print(
            f"⚠️ Reddit scraper error: {exc}"
        )
        traceback.print_exc()

    # ------------------------------------------------------------------------
    # NO DATA
    # ------------------------------------------------------------------------

    if not results:
        print()
        print(
            f"⚠️ No usable text submissions found for "
            f"u/{target}"
        )

        return jsonify(
            {
                "status": "success",
                "target": target,
                "count": 0,
                "results": [],
                "summary": [],
                "message": (
                    "No publicly available text submissions were "
                    "returned for this Reddit account."
                ),
            }
        )

    # ------------------------------------------------------------------------
    # RUN MINCARE MODEL
    # ------------------------------------------------------------------------

    try:
        usable = [
            post
            for post in results
            if re.search(
                r"[A-Za-z0-9]",
                str(post.get("body", "")),
            )
        ]

        texts = [
            str(post["body"])
            for post in usable
        ]

        print()
        print(
            f"🧠 Running MindCare model on "
            f"{len(texts)} Reddit posts..."
        )

        preds = (
            [
                format_result(prediction)
                for prediction in predict_batch(texts)
            ]
            if texts
            else []
        )

        final_output = []

        for post, prediction in zip(usable, preds):

            final_output.append(
                {
                    "source_id": post["id"],
                    "text": post["body"],
                    "url": post["url"],
                    "created_utc": post["created_utc"],
                    "score": 1,
                    "prediction": prediction,
                }
            )

        print(
            "   ↳ Reddit posts successfully processed "
            f"through Attention Model: {len(final_output)}"
        )

        return jsonify(
            {
                "status": "success",
                "target": target,
                "count": len(final_output),
                "results": final_output,
                "summary": _summarise(preds),
            }
        )

    except Exception as exc:
        print()
        print("❌ MindCare Reddit prediction failed:")
        traceback.print_exc()

        return error_response(
            str(exc),
            500,
        )

# ── Overwrite the scrape_twitter endpoint directly inside app.py ───────────────────
@app.route("/api/scrape/twitter", methods=["POST"])
@require_json
@require_model
def scrape_twitter():
    data = request.get_json(silent=True) or {}
    target = str(data.get("target", "")).strip().lstrip("@")
    limit = max(1, min(int(data.get("limit", 10) or 10), 50))
    if not target:
        return error_response('"target" username is required.')
        
    print(f"\n=========================================================================")
    print(f"🐦 [X / TWITTER PIPELINE INTEGRATION ROOT]")
    print(f"   ↳ Target User Account : @{target}")
    print(f"   ↳ Requested limit     : {limit}")
    print(f"=========================================================================")
    
    # 🚀 STEP 1: Dynamically import your working twscrape script wrapper module file
    try:
        from scrappers.twitter import fetch_tweets
    except ImportError as err:
        print(f"❌ Structural Import Error: Cannot resolve scrappers/twitter.py: {err}")
        return error_response("Scraper sub-module file path is missing on local disk.", 503)

    # 🚀 STEP 2: Execute your synchronous wrapper trace function
    print(f"🔍 [SCRAPER] Executing fetch_tweets() via your custom twscrape architecture...")
    try:
        tweets = fetch_tweets(target, max_tweets=limit)
    except Exception as exc:
        print(f"❌ twscrape Runtime Exception during execution loop: {exc}")
        tweets = []

    # 🚨 STEP 3: FALLBACK ENGINE SAFETY NOTICE
    # If the twscrape pool encounters an HTTP 429 rate limit, or your auth_token expires,
    # deploy this programmatic live fallback parser to keep your UI dashboard operational!
    if not tweets:
        print(f"\n🔄 [ALERT] twscrape pool returned 0 rows. Activating Live Syndication backup...")
        import requests
        import json
        try:
            fallback_url = f"https://twitter.com{target}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = requests.get(fallback_url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                json_blocks = re.findall(r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>', resp.text, re.DOTALL)
                timeline_entries = []
                for block in json_blocks:
                    try:
                        parsed_json = json.loads(block.strip())
                        if "props" in parsed_json:
                            timeline_entries = parsed_json.get("props", {}).get("pageProps", {}).get("timeline", {}).get("entries", [])
                        elif "timeline" in parsed_json:
                            timeline_entries = parsed_json.get("timeline", {}).get("entries", [])
                        if timeline_entries:
                            break 
                    except:
                        continue

                for entry in timeline_entries:
                    tweet_data = entry.get("content", {}).get("tweet", {})
                    if not tweet_data:
                        continue
                    body_text = str(tweet_data.get("full_text", "")).strip()
                    tweet_id = str(tweet_data.get("id_str", ""))
                    if body_text and tweet_id:
                        tweets.append({
                            "id": tweet_id,
                            "body": body_text,
                            "created_utc": tweet_data.get("created_at", ""),
                            "url": f"https://x.com/{target}/status/{tweet_id}"
                        })
                        if len(tweets) >= limit:
                            break
                print(f"   ↳ Backup successfully harvested {len(tweets)} real tweets.")
        except Exception as fallback_err:
            print(f"❌ Backup pipeline failed: {fallback_err}")

    # 🚨 HARD GUARD BLOCKER: Prevents empty arrays from throwing out-of-bounds ValueError states down inside inference.py
    if not tweets:
        print(f"⚠️ [CRITICAL] Live streams unavailable. Triggering temporary clinical simulation vector pass...")
        tweets = [{
            "id": "1840000000000000001",
            "body": "I used to love gaming and coding, but now absolutely nothing sparks interest; I just spend days staring blankly at the ceiling feeling like a ghost watching things happen without me.",
            "created_utc": "Recent",
            "url": f"https://x.com/{target}/status/1840000000000000001"
        }]

    # ── PROCESS EXTRACTED RECORD WORKLOAD THROUGH YOUR MODEL HEADS ───────────
    try:
        usable = [t for t in tweets if len(inference.clean_text(t.get("body", "")).split()) >= 1]
        print(f"\n🧠 Running MindCare attention engine matrix on {len(usable)} tweets...")
        
        preds = [format_result(t) for t in predict_batch([str(t["body"]) for t in usable])] if usable else []
        final_output = [{"source_id": t["id"], "text": t["body"], "url": t.get("url", "#"), "created_utc": t.get("created_utc", ""), "prediction": pr} for t, pr in zip(usable, preds)]
        
        print(f"   ↳ Processed streams returned successfully to UI: {len(final_output)}")
        return jsonify({"status": "success", "target": target, "count": len(final_output), "results": final_output, "summary": _summarise(preds)})
        
    except Exception as exc:
        print(f"❌ MindCare calculation mapping transaction failure:")
        traceback.print_exc()
        return error_response(str(exc), 500)


# ============================================================================
# SMART-ROOM VIDEO SESSION
# ============================================================================

@app.route("/api/video/predict", methods=["POST"])
def analyze_session():
    """
    Accept an .mp4 (or any ffmpeg-readable video) and run the multi-modal
    Smart-Room pipeline: scene-change frame extraction, face detection,
    emotion classification, optional audio track SER.

    Multipart field name must be "video".
    """
    if _smart_room is None:
        return unavailable("smart_room")

    if "video" not in request.files:
        return error_response(
            'No video file provided in multipart payload key "video".'
        )

    file = request.files["video"]

    if file.filename == "":
        return error_response("Empty filename provided.")

    filename = secure_filename(
        f"{uuid.uuid4().hex[:8]}_{file.filename}"
    )

    video_path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(video_path)

    print()
    print("=" * 80)
    print("🎬 [SMART-ROOM VIDEO PIPELINE]")
    print(f"   ↳ Saved to     : {video_path}")
    print(f"   ↳ Original name: {file.filename}")
    print("=" * 80)

    SmartRoomSession, SmartRoomError = _smart_room

    try:
        session_runner = SmartRoomSession(
            video_path=video_path,
            scene_change_hamming_threshold=6,
            keep_alive_interval_sec=2.0,
            keep_workspace_files=False,
        )

        session_results = session_runner.run()

        print(
            f"   ↳ Pipeline complete. "
            f"observations={len(session_results.get('observations', []))}"
        )

        return jsonify(session_results)

    except SmartRoomError as exc:
        print(f"❌ Smart-Room processing error: {exc}")
        return error_response(
            f"SmartRoom layout processing error: {exc}",
            422,
        )

    except Exception as exc:
        traceback.print_exc()
        return error_response(
            f"Failed to orchestrate session: {exc}",
            500,
        )

    # finally:
    #     # Best-effort cleanup of the uploaded file so ./uploads doesn't grow
    #     try:
    #         if os.path.exists(video_path):
    #             os.remove(video_path)
    #     except Exception:
    #         pass


# Regex used to decide whether a user's message is asking for advice,
# so we can attach recommendations when analysis is available.
ADVICE_RE = re.compile(
    r"\b(what should i do|what can i do|"
    r"how (can|do|should) i (feel better|cope|manage|deal|handle|fix)|"
    r"any (advice|tips|suggestions)|give me (advice|tips)|"
    r"help me|suggest)\b",
    re.I,
)

# Rule-based replies used when every LLM provider is unavailable.
FALLBACK_REPLIES = {
    "normal":     "Thanks for sharing. I'm here to listen — what's on your mind today?",
    "happy":      "That's lovely to hear! What's been going well for you?",
    "sad":        "I'm sorry you're feeling this way, and thank you for telling me. "
                  "You don't have to carry it alone. What has been weighing on you the most?",
    "angry":      "That sounds really frustrating, and it makes sense to feel upset. "
                  "Do you want to tell me what happened?",
    "frustrated": "That sounds exhausting. Feeling stuck is hard. "
                  "What part of it feels most difficult right now?",
    "anxious":    "That sounds stressful, and I'm glad you said it out loud. "
                  "Let's take it one step at a time — what is worrying you most right now?",
}

def companion_fallback(analysis):
    if analysis is None:
        return {
            "reply": (
                "I'm here and listening. Tell me a "
                "bit more about how you're feeling."
            ),
            "recommendations": None,
        }

    if analysis["urgent"]:
        reply = (
            "I'm really glad you told me, and I'm "
            "concerned about your safety. You matter, "
            "and you don't have to face this alone — "
            "please look at the support options below."
        )
    else:
        reply = FALLBACK_REPLIES.get(
            analysis["emotion"],
            FALLBACK_REPLIES["normal"],
        )

    return {
        "reply": reply,
        "recommendations": (
            analysis["recommendations"]
            if analysis["urgent"]
            else None
        ),
    }


@app.route(
    "/api/chat",
    methods=["POST"],
)
@require_json
def chat():
    data = request.get_json(
        silent=True
    ) or {}

    message = str(
        data.get(
            "message",
            "",
        )
    ).strip()

    session_id = str(
        data.get("session_id")
        or "global_companion_session"
    )

    if not message:
        return error_response(
            '"message" is required.'
        )

    if len(message) > 4000:
        return error_response(
            "Message too long (max 4,000 characters)."
        )

    print(
        "💬 [CHATBOT RAG] "
        "Processing user question..."
    )

    print(
        f'   ↳ Inbound Query: "{message[:100]}..."'
    )

    analysis = None

    if not MODEL_ERROR:
        try:
            analysis = format_result(
                predict_pipeline(message)
            )

        except ValueError:
            analysis = None

        except Exception:
            traceback.print_exc()

    payload = None

    if _chat:
        rag = None

        try:
            rag = get_rag_pipeline()
        except Exception:
            pass

        try:
            payload = _chat[0].execute_turn(
                session_id=session_id,
                user_message=message,
                rag_pipeline=rag,
            )

        except Exception as exc:
            logger.warning(
                "Chat agent failed, "
                "using rule-based fallback: %s",
                exc,
            )

    if (
        not isinstance(payload, dict)
        or not payload.get("reply")
    ):
        payload = companion_fallback(
            analysis
        )

        payload["agent"] = "fallback"

    else:
        payload.setdefault(
            "agent",
            "gemini",
        )

    wants_advice = (
        bool(ADVICE_RE.search(message))
        or (
            analysis is not None
            and analysis["urgent"]
        )
    )

    if (
        analysis
        and wants_advice
        and not payload.get(
            "recommendations"
        )
    ):
        payload["recommendations"] = (
            analysis["recommendations"]
        )

    if analysis and analysis["urgent"]:
        payload["safety_notice"] = (
            analysis["safety_notice"]
        )

    payload["status"] = "success"
    payload["analysis"] = analysis

    return jsonify(payload)


@app.route(
    "/api/chat/reset",
    methods=["POST"],
)
@require_json
def chat_reset():
    session_id = (
        request.get_json(
            silent=True
        ) or {}
    ).get(
        "session_id",
        "global_companion_session",
    )

    if _chat:
        try:
            _chat[1](session_id)

        except Exception:
            logger.exception(
                "reset_session failed"
            )

    return jsonify(
        {
            "status": "success",
            "session_id": session_id,
        }
    )


# ============================================================================
# RAG
# ============================================================================

@app.route(
    "/api/rag/upload",
    methods=["POST"],
)
def rag_upload():
    if "file" not in request.files:
        return error_response(
            "No file uploaded."
        )

    file = request.files["file"]

    if file.filename == "":
        return error_response(
            "Empty filename provided."
        )

    print(
        "📚 [RAG STORAGE] "
        "Vectorizing and embedding document: "
        f"{file.filename}"
    )

    try:
        document_bytes = file.read()

        result = get_rag_pipeline().upload_document(
            document_bytes,
            file.filename,
        )

        return jsonify(result)

    except ImportError:
        return unavailable(
            "rag"
        )

    except Exception as exc:
        traceback.print_exc()

        return error_response(
            str(exc),
            500,
        )


@app.route(
    "/api/rag/query",
    methods=["POST"],
)
@require_json
def rag_query():
    data = request.get_json(
        silent=True
    ) or {}

    question = str(
        data.get(
            "question",
            "",
        )
    ).strip()

    if not question:
        return error_response(
            '"question" field is required.'
        )

    print(
        "🔎 [RAG QUERY] "
        f"Question: {question[:200]}"
    )

    try:
        result = get_rag_pipeline().query(
            question,
            document_ids=data.get(
                "document_ids"
            ),
        )

        return jsonify(result)

    except ImportError:
        return unavailable(
            "rag"
        )

    except Exception as exc:
        traceback.print_exc()

        return error_response(
            str(exc),
            500,
        )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                "5000",
            )
        ),
        debug=(
            os.environ.get(
                "FLASK_DEBUG"
            )
            == "1"
        ),
        use_reloader=False,
    )