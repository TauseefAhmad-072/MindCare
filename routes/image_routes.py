from flask import Blueprint, request, jsonify
from modules.image_emotion.predictor import predict_bytes

# Fixed name mismatch to perfectly resolve app.py linking loop
image_api = Blueprint('image_api', __name__)

@image_api.route('/predict', methods=['POST'])
def predict_image_emotion():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'error': 'No image file uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'Empty filename'}), 400
    try:
        res = predict_bytes(file.read())
        return jsonify({'status': 'success', 'result': res})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
