"""
recommendations.py — supportive, non-diagnostic guidance chosen from a model result.

Single source of truth: the React app only renders what the API returns.

Priority order:
    1. Crisis (safety_notice present)         → crisis resources, urgent=True
    2. Confident disorder pattern (>= 0.75)   → pattern-specific guidance
    3. Emotion-based fallback                 → general supportive tips

Design decisions:
  * The disorder head was trained on a limited subset of the 9 clinical
    patterns it emits. Below 0.75 confidence we do NOT route the user
    toward a specific pattern — we fall back to emotion-based tips.
    This threshold MUST stay in sync with CONFIDENCE_FLOOR in
    inference._trace().
  * Crisis detection always wins. If safety_notice is set, disorder and
    emotion paths are bypassed entirely.
  * This module produces non-diagnostic support language only. It never
    says "you have X" — it says "this resembles descriptions of X" and
    points toward professional support.
"""
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Crisis resources
# ─────────────────────────────────────────────────────────────────────────────

CRISIS_HELP = [
    "Call a crisis line now — Pakistan: 0311-7786264 · Rozan: (051) 2890505 · "
    "elsewhere: findahelpline.com",
    "If you may act on these thoughts or are in immediate danger, go to the "
    "nearest emergency room or call your local emergency number.",
    "Do not stay alone — tell a family member, friend or flatmate how you feel "
    "right now.",
    "Move anything you could use to hurt yourself out of reach while the feeling "
    "passes.",
    "After you are safe, book an appointment with a doctor, psychiatrist or "
    "clinical psychologist.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Emotion-based supportive guidance (the default path)
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_TIPS: Dict[str, dict] = {
    "normal": {
        "title": "Keeping steady",
        "consolation": "Nothing in this text stands out as a concern. Here is "
                       "how to keep your wellbeing strong:",
        "steps": [
            "Keep a regular sleep schedule (7–8 hours).",
            "Move your body for about 30 minutes most days.",
            "Stay in touch with people you trust, even briefly.",
            "Try a two-minute daily check-in: how am I really doing?",
        ],
    },
    "happy": {
        "title": "Making the good last",
        "consolation": "It's lovely to read something positive. A few ways to "
                       "build on it:",
        "steps": [
            "Note what contributed to feeling good today.",
            "Share the good news with someone you care about.",
            "Protect the routines that support you (sleep, food, movement).",
            "Use the energy for something meaningful to you.",
        ],
    },
    "sad": {
        "title": "When things feel heavy",
        "consolation": "What you're carrying sounds heavy, and your feelings "
                       "are valid. You don't have to handle it alone:",
        "steps": [
            "Tell one trusted person how you are feeling.",
            "Pick one small, doable thing today — a walk, a shower, a proper meal.",
            "Keep sleep and meals as regular as you can.",
            "If low mood lasts more than two weeks or deepens, speak to a "
            "counsellor or doctor.",
        ],
    },
    "angry": {
        "title": "Cooling down",
        "consolation": "Anger is a signal that something matters or feels "
                       "unfair. Here is how to handle it safely:",
        "steps": [
            "Pause before responding: breathe in for 4 seconds, out for 6, "
            "five times.",
            "Step away from the trigger for 10–20 minutes.",
            "Release the energy physically — a brisk walk or exercise.",
            "Write down what triggered it, then decide what response you "
            "actually want.",
        ],
    },
    "frustrated": {
        "title": "Getting unstuck",
        "consolation": "Feeling stuck is exhausting. Small steps usually help "
                       "more than pushing harder:",
        "steps": [
            "Break the problem into the smallest next step and do only that.",
            "Take a short break away from the screen or task.",
            "Ask someone for help or a fresh perspective.",
            "Lower the bar for today — 'good enough' counts.",
        ],
    },
    "anxious": {
        "title": "Calming anxiety",
        "consolation": "Anxiety is very common and it can ease. These steps "
                       "help many people:",
        "steps": [
            "Slow breathing: in for 4, hold 4, out for 6, for five minutes.",
            "Ground yourself with 5-4-3-2-1: five things you see, four you "
            "feel, three you hear…",
            "Write your worries down and mark which ones you can act on today.",
            "Limit caffeine and doom-scrolling; if anxiety is constant, talk "
            "to a professional.",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Disorder-pattern guidance (only fires above the confidence floor)
# ─────────────────────────────────────────────────────────────────────────────

DISORDER_TIPS: Dict[str, dict] = {
    "adhd": {
        "title": "Attention & focus patterns",
        "consolation": "Your text resembles how people describe attention and "
                       "executive-function struggles. This is not a diagnosis, "
                       "but these strategies help:",
        "steps": [
            "Externalise memory: timers, visible to-do lists, calendar alerts.",
            "Work in 10–25 minute blocks with short breaks.",
            "Silence notifications and try body-doubling (working alongside "
            "someone).",
            "Consider an assessment through your doctor or campus support "
            "services.",
        ],
    },
    "anxiety_disorder": {
        "title": "Anxiety patterns",
        "consolation": "Your text resembles descriptions of persistent anxiety. "
                       "It is treatable, and you can feel better:",
        "steps": [
            "Practise slow diaphragmatic breathing daily, not only in a crisis.",
            "Try a thought record: the worry, the evidence for it, a balanced "
            "alternative.",
            "Keep a steady sleep routine and cut back on caffeine.",
            "Ask a professional about CBT, which has strong evidence for "
            "anxiety.",
        ],
    },
    "aspergers": {
        "title": "Autism-spectrum patterns",
        "consolation": "Your text resembles how autistic people describe social "
                       "and sensory experiences. This is not a diagnosis:",
        "steps": [
            "Build predictable routines and give yourself recovery time after "
            "social effort.",
            "Plan sensory breaks (quiet space, headphones, dim light).",
            "Ask campus disability or accessibility services about adjustments.",
            "If you want clarity, request an assessment from a clinician "
            "experienced in autism in adults.",
        ],
    },
    "bipolar": {
        "title": "Mood-swing patterns",
        "consolation": "Your text resembles descriptions of big swings in "
                       "energy and mood. A professional can help make sense "
                       "of this:",
        "steps": [
            "Track sleep, mood and energy daily for two weeks.",
            "Keep sleep times as regular as possible — sleep loss can trigger "
            "highs.",
            "Avoid big financial or life decisions at the peak or bottom of a "
            "swing.",
            "Book a psychiatrist or doctor visit and share your tracking notes.",
        ],
    },
    "bpd": {
        "title": "Intense-emotion patterns",
        "consolation": "Your text resembles descriptions of very intense "
                       "emotions and fear of abandonment. These feelings are "
                       "real and treatable:",
        "steps": [
            "When emotions spike, delay acting for 20 minutes and use a "
            "calming skill (cold-free options: paced breathing, music, a walk).",
            "Name the feeling, then name the fact behind it.",
            "Lean on one steady, safe person during hard moments.",
            "Ask a clinician about Dialectical Behaviour Therapy (DBT).",
        ],
    },
    "depression": {
        "title": "Low-mood patterns",
        "consolation": "Your text resembles descriptions of persistent low "
                       "mood and emptiness. You deserve support, and depression "
                       "responds well to treatment:",
        "steps": [
            "Tell someone you trust how long this has been going on.",
            "Do one small activity that used to matter, even for five minutes.",
            "Keep a regular sleep, meal and daylight routine.",
            "Book a doctor or counsellor visit — therapy and medication both "
            "help.",
        ],
    },
    "ocd": {
        "title": "Intrusive-thought & ritual patterns",
        "consolation": "Your text resembles descriptions of intrusive thoughts "
                       "and rituals. This is common, and effective treatment "
                       "exists:",
        "steps": [
            "Notice the urge to check or repeat, and try to delay it a little "
            "longer each time.",
            "Avoid seeking repeated reassurance — it feeds the cycle.",
            "Remember: an intrusive thought is not a wish or a prediction.",
            "Ask a professional about Exposure and Response Prevention (ERP).",
        ],
    },
    "ptsd": {
        "title": "Trauma-reaction patterns",
        "consolation": "Your text resembles descriptions of trauma reactions "
                       "such as flashbacks and being on edge. Healing is "
                       "possible:",
        "steps": [
            "During a flashback, ground yourself: feel your feet, name five "
            "things you can see, remind yourself it is now, not then.",
            "Keep routines and safe places that help you feel secure.",
            "Avoid numbing with alcohol or constant avoidance.",
            "Look for trauma-focused therapy (e.g. EMDR or trauma-focused CBT).",
        ],
    },
    "schizophrenia": {
        "title": "Unusual-perception patterns",
        "consolation": "Your text resembles descriptions of unusual perceptions "
                       "or confused thinking. Please involve a professional soon:",
        "steps": [
            "Speak to a doctor or psychiatrist soon about what you are "
            "experiencing.",
            "Tell someone you trust so you are not facing it alone.",
            "Keep sleep regular and avoid alcohol or drugs, which can worsen "
            "symptoms.",
            "If you feel unsafe or cannot tell what is real, go to an emergency "
            "department.",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Minimum top-disorder probability required before we route the user to
# pattern-specific advice. Must stay in sync with CONFIDENCE_FLOOR in
# inference._trace(). Below this, we fall back to emotion-based support.
TOP_DISORDER_THRESHOLD = 0.75


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def recommendations_for(result: dict) -> Optional[dict]:
    """
    Choose a supportive guidance block for a formatted API result.

    `result` is the dict produced by format_result() in app.py, which
    contains at minimum:
        - safety_notice       : str | None
        - top_disorder        : {"name": str, "probability": float} | None
        - emotion             : str (one of EMOTION_TIPS keys)

    Returns a dict with keys: key, title, consolation, steps, urgent.
    Returns None if nothing applies.
    """

    # ── 1. Crisis path — always wins ─────────────────────────────────────
    if result.get("safety_notice"):
        return {
            "key": "crisis",
            "title": "Please reach out for support now",
            "consolation": (
                "I'm concerned about your safety. You are not alone, and help "
                "is available right now:"
            ),
            "steps": CRISIS_HELP,
            "urgent": True,
        }

    # ── 2. Confident disorder pattern ────────────────────────────────────
    top = result.get("top_disorder")
    if (
        top
        and isinstance(top, dict)
        and top.get("name") in DISORDER_TIPS
        and float(top.get("probability", 0.0)) >= TOP_DISORDER_THRESHOLD
    ):
        return {
            "key": top["name"],
            "urgent": False,
            **DISORDER_TIPS[top["name"]],
        }

    # ── 3. Emotion-based fallback ────────────────────────────────────────
    emotion = result.get("emotion")
    if emotion and emotion in EMOTION_TIPS:
        return {
            "key": emotion,
            "urgent": False,
            **EMOTION_TIPS[emotion],
        }

    return None


__all__ = ["recommendations_for", "CRISIS_HELP", "EMOTION_TIPS",
           "DISORDER_TIPS", "TOP_DISORDER_THRESHOLD"]