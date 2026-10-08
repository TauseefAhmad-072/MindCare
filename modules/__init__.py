"""
MindCare AI Module Suite — standalone, independently-testable ML modules.

  modules.text           MODULE A — Text Mental-Health Analysis (reuses
                          the existing trained MindCarePredictor)
  modules.image_emotion   MODULE B — Image Emotion Recognition (pretrained ViT)
  modules.audio_emotion   MODULE C — Audio/Speech Emotion Recognition
                          (pretrained Wav2Vec2, heuristic fallback)
  modules.rag              MODULE D — RAG over patient reports (Phase 3)

Each module exposes a small, dependency-free public function (see each
package's __init__.py) so the future Flask/React app can call:

    from modules.text import predict as text_predict
    from modules.image_emotion import predict as image_predict
    from modules.audio_emotion import predict as audio_predict
"""
