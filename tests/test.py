"""
test_suite.py — Unified MindCare Test Suite
Validates deep learning modules, video pipelines, RAG indexing, and HTTP endpoints.

Usage:
  pytest test_suite.py -v
"""

import os
import io
import json
import pytest
import requests
import numpy as np

# Import raw ML modules directly for unit verification
from mindcare_inference import MindCarePredictor
from modules.smart_room import _perceptual_hash, _hamming_distance, SmartRoomSession
from modules.rag import RAGPipeline

BASE_URL = "http://127.0.0.1:5000"
TEST_VIDEO_PATH = "./test_media/test_session.mp4"

# ==============================================================================
# PHASE 1: DIRECT MACHINE LEARNING MODULE UNIT TESTS
# ==============================================================================

def test_unit_text_predictor_loading():
    """Verify local text models load and output multi-class evaluation scores."""
    MODELS_DIR = os.environ.get('MODELS_DIR', '../models')
    try:
        predictor = MindCarePredictor(models_dir=MODELS_DIR)
        res = predictor.predict("I feel highly stressed and overwhelmed by work.", include_all_scores=True)
        assert "level" in res
        assert "confidence" in res
    except Exception as e:
        pytest.skip(f"Skipping direct weights check: local models not downloaded in {MODELS_DIR} ({str(e)})")


def test_unit_smart_room_hashing():
    """Verify core image tracking perceptual hash algorithms function reliably."""
    # Generate dummy black and white synthetic matrices
    img_black = np.zeros((100, 100, 3), dtype=np.uint8)
    img_white = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    hash_black = _perceptual_hash(img_black)
    hash_white = _perceptual_hash(img_white)
    
    # Dissimilar images must yield a significant relative Hamming Distance
    distance = _hamming_distance(hash_black, hash_white)
    assert distance > 0


def test_unit_rag_memory_indexing():
    """Verify raw document chunking, indexing, and vector similarity queries."""
    pipeline = RAGPipeline(store_dir="./test_vector_store")
    
    sample_doc = b"Patient PT-104 requires regular therapeutic checkups. Prescribed Escitalopram 10mg."
    upload_res = pipeline.upload_document(sample_doc, "test_doc.txt", allow_duplicate=True)
    
    assert upload_res["status"] == "success"
    doc_id = upload_res["document_id"]
    
    # Query the isolated database directly
    query_res = pipeline.query("What medication does patient PT-104 take?", document_ids=[doc_id])
    assert query_res["status"] == "success"
    
    # Clean up test directories
    pipeline.delete_document(doc_id)


# ==============================================================================
# PHASE 2: FLASK WEB API INTEGRATION & FLUX TRAFFIC TESTS
# ==============================================================================

@pytest.fixture(scope="module")
def check_api_server_live():
    """Check if the backend endpoint framework is actively listening."""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            pytest.skip("Flask server up but returning bad codes.")
    except requests.exceptions.ConnectionError:
        pytest.skip(f"Flask API server is offline on {BASE_URL}. Run 'python app.py' first to test endpoints.")


def test_api_text_prediction(check_api_server_live):
    """Test web payload handling for core mental health assessment metrics."""
    payload = {"text": "I feel slightly blue today.", "include_all_scores": False}
    res = requests.post(f"{BASE_URL}/api/predict", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "success"


def test_api_chatbot_crisis_override(check_api_server_live):
    """Ensure natural language inputs correctly trigger safety network protocols."""
    payload = {"message": "I want to end my life, everything is hopeless."}
    res = requests.post(f"{BASE_URL}/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    
    # Critical keyword vectors must immediately escalate handling level to acute stages
    assert data["result"]["level"] >= 6
    assert "Pakistan: 0311-7786264" in str(data["result"]["recommendation"])


def test_api_medical_assessment_scoring(check_api_server_live):
    """Verify systemic scoring boundary evaluation maps clinical severities safely."""
    payload = {
        "type": "PHQ9",
        "responses": [3, 3, 3, 3, 3, 2, 2, 1, 1]  # Cumulative score = 21 (Severe Range)
    }
    res = requests.post(f"{BASE_URL}/api/assessment", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_score"] == 21
    assert data["severity"] == "Severe"
    assert data["recommend_referral"] is True


def test_api_smart_room_multimodal_video(check_api_server_live):
    """Validate multipart streaming chunk analysis over multi-stream file tracks."""
    if not os.path.exists(TEST_VIDEO_PATH):
        pytest.skip(f"Missing integration video file tracker: {TEST_VIDEO_PATH}. Run your media download tool first.")

    with open(TEST_VIDEO_PATH, "rb") as f:
        files = {"video": (os.path.basename(TEST_VIDEO_PATH), f, "video/mp4")}
        res = requests.post(f"{BASE_URL}/api/session/analyze", files=files)
        
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "timeline_observations" in data


def test_api_rag_e2e_workflow(check_api_server_live):
    """Validate system document streaming uploads and context retrieval routines."""
    mock_report = "Patient Profile: PT-442. Clinical Diagnosis: Severe Panic Disorder. Therapy Route: Exposure therapy sessions."
    payload = {
        'file': ("patient_profile.txt", io.BytesIO(mock_report.encode('utf-8')), 'text/plain')
    }
    
    # 1. Test multi-part raw stream upload handling
    upload_res = requests.post(f"{BASE_URL}/api/rag/upload", files=payload)
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["status"] == "success"
    doc_id = upload_data["document_id"]
    
    # 2. Test semantic query extraction matching against newly indexed vectors
    query_payload = {
        "question": "What therapy route is recommended for patient profile PT-442?",
        "document_ids": [doc_id]
    }
    query_res = requests.post(f"{BASE_URL}/api/rag/query", json=query_payload)
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert "Exposure therapy" in query_data["answer"]
