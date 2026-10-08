"""
tests/test_scraper_inference.py — Subfolder-Safe Local Evaluation Script
─────────────────────────────────────────────────────────────────────────────
Fixed path parameters to step out of the tests/ folder and tap the root project directory,
allowing error-free imports of mindcare_inference and local checkpoint assets.
"""
import os
import sys
import json

# FIXED: Appends the parent folder (root project directory) to the active execution path environment
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Safe programmatic import of the root inference script now that pathing is bound
from mindcare_inference import MindCarePredictor

def run_local_simulation():
    print("=" * 70)
    print("🚀 INITIALIZING MINDCARE LOCAL MODEL TEST HARNESS (TESTS SUBFOLDER MODE)")
    print("=" * 70)

    # FIXED: Re-directs the models directory lookups straight to the parent root folder path tracking grid
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Error: Checkpoint directory missing at absolute location path: {MODELS_DIR}")
        return

    try:
        predictor = MindCarePredictor(models_dir=MODELS_DIR)
    except Exception as e:
        print(f"❌ Critical error loading your ensemble model checkpoints: {str(e)}")
        return

    # Array of simulated metrics reflecting clinical states
    mock_scraped_feeds = [
        {
            "platform": "Reddit Local Sandbox Mode",
            "author": "Simulated_User_High_Risk",
            "text": "Everything feels completely hopeless right now. I feel like an absolute burden to my family, and I'm trapped in a constant cycle of empty numbness. I can’t stop crying randomly during the day."
        },
        {
            "platform": "X / Twitter Local Sandbox Mode",
            "author": "Simulated_User_Burnout",
            "text": "I am so mentally exhausted from this semester. The deadlines are stacking up and I feel constantly overwhelmed trying to keep my head above water. Struggling to sleep properly."
        },
        {
            "platform": "Reddit Local Sandbox Mode",
            "author": "Simulated_User_Normal",
            "text": "Had a really productive day finishing up the backend architecture layouts! Looking forward to relaxing this weekend and hanging out with the team."
        }
    ]

    print("\n📊 RUNNING ENSEMBLE INFERENCE OVER MOCK DATA ARRAYS...\n")

    for idx, post in enumerate(mock_scraped_feeds, 1):
        print(f"[{idx}] Scanning Feed Target: {post['author']} ({post['platform']})")
        print(f"📝 Text: \"{post['text']}\"")
        
        try:
            # Execute predictions directly through your SBERT + BiLSTM heads
            prediction = predictor.predict(post["text"])
            
            print(f"🔮 Telemetry Response Verdict: Level {prediction['level']} — {prediction['level_name']}")
            print(f"🎯 Classifier Certainty Metric: {round(prediction['confidence'] * 100, 2)}%")
            print(f"💡 System Recommended Intervention: {prediction['intervention']}")
            if prediction.get('highlights'):
                print(f"🖍️ Active Keyword Highlights Engaged: { [h['word'] for h in prediction['highlights']] }")
        except Exception as eval_err:
            print(f"❌ Inference loop failed on current index: {str(eval_err)}")
            
        print("─" * 70)

if __name__ == "__main__":
    run_local_simulation()
