"""
routes/audio_routes.py — Production Blueprint for Module C (Speech SER)
Directly maps multipart data stream arrays to core binary SER parsers.
"""
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from modules.audio_emotion.predictor import predict_bytes

logger = logging.getLogger("mindcare.api.audio")

# Define a clean, single absolute blueprint instance variable
audio_api = Blueprint('audio_api', __name__)

@audio_api.route('/predict', methods=['POST'])
def predict_audio_emotion():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'error': 'No audio file uploaded under multipart key "audio"'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'Empty filename uploaded'}), 400

    try:
        # Read file stream bytes directly at the route controller layer
        raw_bytes = file.read()
        
        # Pass bytes straight into the high-performance binary predictor
        res = predict_bytes(raw_bytes, filename_hint=file.filename)
        
        # Return cleanly back to the React UI context with both top-level and nested structure
        payload = {
            'status': 'success',
            'result': res,
            'emotion': res.get('emotion', 'neutral'),
            'dominant_emotion': res.get('dominant_emotion', 'neutral'),
            'confidence': res.get('confidence', 0.0),
            'probabilities': res.get('probabilities', {}),
            'emotion_distribution': res.get('emotion_distribution', {}),
            'waveform_analysis': res.get('waveform_analysis', {}),
            'waveform_tone_analysis': res.get('waveform_tone_analysis', {}),
            'tone_explanation': res.get('tone_explanation', ''),
            'duration': res.get('duration', 0.0),
            'captured_text': res.get('captured_text', ''),
            'normalized_text': res.get('normalized_text', ''),
            'urdu_sentiment': res.get('urdu_sentiment'),
            'safety_hit': res.get('safety_hit', False),
            'safety_notice': res.get('safety_notice'),
            'audio_disclaimer': res.get('audio_disclaimer') or res.get('disclaimer')
        }
        return jsonify(payload)
    except Exception as e:
        logger.exception("Audio SER runtime blueprint inference exception thrown")
        return jsonify({'status': 'error', 'error': f"Audio stream inference failed: {str(e)}"}), 500
