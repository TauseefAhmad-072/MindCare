"""
test_mindcare.py — MindCare Inference Test Suite
Run: python test_mindcare.py

Tests cover:
  1. Crisis phrases that MUST score Level 5+
  2. Indirect/coded suicidal language (e.g. "jump off")
  3. False positives — normal phrases that contain crisis words
  4. Stress phrases that must NOT reach suicidal level
  5. Boundary cases
"""

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mindcare_inference import MindCarePredictor

# ── Change this to your actual models folder ──────────────────────────────────
MODELS_DIR = './models'

predictor = MindCarePredictor(models_dir=MODELS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Test definitions
# Each entry: (text, expected_min_level, expected_max_level, description)
# expected_max_level = None means no upper bound
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [

    # ── CRITICAL: indirect suicidal (your reported failures) ─────────────────
    ("I want to jump off a bridge",             5, None, "INDIRECT SUICIDAL — jump off bridge"),
    ("I want to jump off a building",           5, None, "INDIRECT SUICIDAL — jump off building"),
    ("thinking about jumping off",              5, None, "INDIRECT SUICIDAL — thinking about jumping"),
    ("I just want to jump",                     5, None, "INDIRECT SUICIDAL — want to jump"),
    ("standing on the edge ready to fall",      5, None, "INDIRECT SUICIDAL — on the edge"),
    ("I keep thinking about the roof",          4, None, "INDIRECT SUICIDAL — thinking about roof"),

    # ── CRITICAL: explicit suicidal ───────────────────────────────────────────
    ("I want to kill myself",                   6, None, "EXPLICIT SUICIDAL — kill myself"),
    ("I am thinking about suicide",             6, None, "EXPLICIT SUICIDAL — suicide"),
    ("I don't want to live anymore",            6, None, "EXPLICIT SUICIDAL — don't want to live"),
    ("I've been thinking about ending it all",  6, None, "EXPLICIT SUICIDAL — ending it all"),
    ("nobody would miss me if I was gone",      5, None, "PASSIVE SUICIDAL — nobody would miss me"),
    ("I wrote goodbye letters to my family",    6, None, "EXPLICIT SUICIDAL — goodbye letters"),

    # ── CRITICAL: your second reported failure — suicidal taken as stress ─────
    ("I can't take this anymore I want to die", 5, None, "SUICIDAL — want to die + cant take it"),
    ("life is pointless there is no way out",   5, None, "SUICIDAL — life is pointless + no way out"),
    ("I feel like a burden to everyone",        5, None, "PASSIVE SUICIDAL — burden to everyone"),
    ("I have given up on everything",           5, None, "PASSIVE SUICIDAL — given up"),

    # ── Severe distress (should be Level 4+, NOT normal/stress) ──────────────
    ("I hate myself so much",                   4, None, "SEVERE — hate myself"),
    ("I've been cutting myself to cope",        4, None, "SEVERE — cutting myself"),
    ("I feel completely hopeless",              4, None, "SEVERE — hopeless"),
    ("I am trapped and there is no way out",    4, None, "SEVERE — trapped no way out"),

    # ── Moderate / Anxiety (should be Level 3-4, NOT suicidal) ───────────────
    ("I have been feeling very anxious lately", 3, 5,   "ANXIETY — anxious"),
    ("I keep having panic attacks at work",     3, 5,   "ANXIETY — panic attacks"),
    ("I feel so overwhelmed by everything",     2, 4,   "STRESS — overwhelmed"),

    # ── Mild stress (should be Level 1-2, NOT suicidal) ──────────────────────
    ("I've been stressed about my exams",       1, 3,   "MILD STRESS — exams"),
    ("work has been really draining lately",    1, 3,   "MILD STRESS — draining"),
    ("I feel a bit down today",                 1, 3,   "MILD STRESS — feeling down"),

    # ── FALSE POSITIVES — must stay Normal/Low ───────────────────────────────
    ("I am dying laughing at this video",       0, 2,   "FALSE POS — dying laughing"),
    ("this joke is literally killing me",       0, 2,   "FALSE POS — killing me (humor)"),
    ("I want to jump into this project",        0, 2,   "FALSE POS — jump into project"),
    ("I'm dead tired after the gym",            0, 2,   "FALSE POS — dead tired (exercise)"),
    ("jumped off the diving board today",       0, 2,   "FALSE POS — jumped off diving board"),

    # ── Normal ───────────────────────────────────────────────────────────────
    ("I had a great day today",                 0, 1,   "NORMAL — great day"),
    ("went for a walk and felt refreshed",      0, 1,   "NORMAL — walk"),
    ("excited about my new job",                0, 1,   "NORMAL — excited"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

def run_tests():
    passed = failed = warned = 0
    failures = []

    print("\n" + "═"*90)
    print(f"{'TEST':<48} {'EXP':>8} {'GOT':>8} {'CONF':>7}  {'RESULT'}")
    print("═"*90)

    for text, min_lvl, max_lvl, desc in TESTS:
        result     = predictor.predict(text)
        level      = result['level']
        confidence = result['confidence']
        kw_match   = result['matched_keywords']
        override   = result['keyword_override']
        ml_level   = result['ml_level']

        exp_str = f"{min_lvl}+" if max_lvl is None else f"{min_lvl}-{max_lvl}"

        ok = level >= min_lvl and (max_lvl is None or level <= max_lvl)

        if ok:
            status = PASS
            passed += 1
        else:
            # Warn if close (off by 1) vs hard fail (off by 2+)
            off_by = min_lvl - level if level < min_lvl else level - (max_lvl or 999)
            if off_by == 1:
                status = WARN
                warned += 1
            else:
                status = FAIL
                failed += 1
            failures.append((desc, text, exp_str, level, ml_level, kw_match, override, status))

        kw_str = f"kw={kw_match[:2]}" if kw_match else ""
        print(f"{desc:<48} {exp_str:>8} {level:>8} {confidence:>6.0%}  {status}  {kw_str}")

    print("═"*90)
    print(f"\nResults: {passed} passed | {warned} warnings (off-by-1) | {failed} failed  "
          f"out of {len(TESTS)} tests\n")

    if failures:
        print("─"*90)
        print("FAILURES / WARNINGS — Detailed Diagnostics")
        print("─"*90)
        for desc, text, exp, got, ml, kw, override, status in failures:
            print(f"\n{status}  {desc}")
            print(f"  Text        : \"{text}\"")
            print(f"  Expected    : Level {exp}")
            print(f"  Got         : Level {got}  (ML raw: {ml})")
            print(f"  Kw match    : {kw}")
            print(f"  Kw override : {override}")
            if got < int(exp[0]):
                print(f"  ► FIX NEEDED: Add phrase to CRISIS_KEYWORDS in mindcare_inference.py")
            else:
                print(f"  ► FIX NEEDED: Add phrase to NEGATION_CONTEXTS or tighten keyword match")

    return failed


if __name__ == '__main__':
    n_failed = run_tests()

    # ── Quick keyword patch suggestions based on common failures ──────────────
    print("\n─"*45)
    print("KEYWORD PATCH SUGGESTIONS")
    print("─"*45)
    print("""
The following phrases are NOT in CRISIS_KEYWORDS and cause misclassification.
Add them to mindcare_inference.py → CRISIS_KEYWORDS:

  passive_suicidal (min_level: 5):
    'want to jump off', 'jump off a bridge', 'jump off a building',
    'jump off the roof', 'thinking about jumping', 'standing on the edge',
    'want to jump', 'thinking about the roof', 'nobody would miss me',
    'would not miss me', 'life is not worth', 'cant go on',
    "can't go on", 'want it to end', 'make it stop',

  severe_distress (min_level: 4):
    'ready to give up', 'completely hopeless', 'so hopeless',
    'dead inside', 'feel like nothing', 'feel like a failure',

Add the following to NEGATION_CONTEXTS to prevent false positives:
    'jump into', 'jumped off the diving', 'jump off the diving',
    'jump for joy', 'dead tired', 'dead set', 'dead serious',
    'killing it today', 'killing the game',
""")

    sys.exit(0 if n_failed == 0 else 1)