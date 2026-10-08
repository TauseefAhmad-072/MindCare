# MindCare — ML Pipeline & Web API

**Semantic & Temporal Analysis of Social Media Text for Student Mental Health Assessment**  
FAST-NUCES FYP | Muhammad Tauseef Ahmad · Ahyan Ali Khan · Hamza Khurram  
Supervised by Mr. Majid Hussain

---

## Project Structure

```
mindcare_ml/
├── notebooks/
│   └── mindcare_train.ipynb     ← Full training pipeline (run this first)
├── api/
│   ├── mindcare_inference.py    ← Inference module (loaded by Flask)
│   └── app.py                   ← Flask REST API server
├── models/                      ← Saved artefacts (created after training)
│   ├── sbert_lr_model.pkl
│   ├── bilstm_attention_best.pt
│   ├── sbert_bilstm_ensemble_best.pt
│   ├── vocab.json
│   └── metadata.json
├── MindCareApp.jsx              ← React web app (paste into Claude artifact)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Step 1 — Train the Models

Open `notebooks/mindcare_train.ipynb` in **Jupyter** or **Google Colab**.

### Data options
```python
# Option A: Kaggle (Mental_Health_.ipynb approach)
import kagglehub
path = kagglehub.dataset_download("suchintikasarkar/sentiment-analysis-for-mental-health")
df   = pd.read_csv(os.path.join(path, "Combined Data.csv"))

# Option B: Your local processed_data.csv (MindCareFYP approach)
df = pd.read_csv("/path/to/processed_data.csv")  # columns: clean_text, label
```

The notebook will:
1. Clean and preprocess text
2. Map labels to MindCare's 8-level severity framework
3. Train **4 models** in sequence:
   - Model 1: SBERT + Logistic Regression (fast baseline)
   - Model 2: LSTM (learned embeddings)
   - Model 3: BiLSTM + Self-Attention
   - Model 4: **SBERT + BiLSTM Ensemble** (primary, targets >92% acc.)
4. Generate SHAP/LIME explanations
5. Save all artefacts to `../models/`

Expected training time (GPU):
- Model 1: ~3 min  
- Model 2: ~8 min  
- Model 3: ~15 min  
- Model 4: ~20 min  

---

## Step 2 — Run the Flask API

```bash
cd api/
pip install -r ../requirements.txt
python app.py
# → Running on http://0.0.0.0:5000
```

### API Endpoints

| Method | Endpoint              | Description                        |
|--------|-----------------------|------------------------------------|
| GET    | `/api/health`         | Health check                       |
| GET    | `/api/classes`        | Get 8-level classification schema  |
| POST   | `/api/predict`        | Predict from text                  |
| POST   | `/api/predict/batch`  | Batch predict (up to 50 texts)     |
| POST   | `/api/assessment`     | PHQ-9 / GAD-7 clinical scoring     |

### Example — predict

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so overwhelmed I cannot sleep and everything is falling apart"}'
```

Response:
```json
{
  "status": "success",
  "result": {
    "level": 3,
    "level_name": "Severe Stress",
    "confidence": 0.874,
    "intervention": "Counselling referral advised",
    "color": "#F97316",
    "icon": "🔶",
    "class_probs": { "Normal": 0.02, "Mild Stress": 0.05, ... },
    "highlights": [
      {"word": "overwhelmed", "score": 0.6, "type": "moderate"},
      {"word": "falling apart", "score": 0.7, "type": "severe"}
    ],
    "model_used": "SBERT + BiLSTM Ensemble",
    "urgent": false
  }
}
```

### Example — PHQ-9

```bash
curl -X POST http://localhost:5000/api/assessment \
  -H "Content-Type: application/json" \
  -d '{"type": "PHQ9", "responses": [2,3,2,3,1,2,2,1,0]}'
```

---

## Step 3 — Web App Integration

### React (Recommended)

Paste the contents of `MindCareApp.jsx` into a new Claude artifact or your React project:

```jsx
// Change API_BASE at the top of MindCareApp.jsx to your deployed URL:
const API_BASE = "https://your-api-domain.com/api";
```

The app includes:
- Text analysis with highlighted keywords (SHAP/LIME visual output)
- 8-level risk gauge with animated probability bars
- PHQ-9 & GAD-7 interactive assessment modals
- Session history with temporal trend chart
- Crisis alert UI for Level 6-7 detections
- Auto-fallback demo mode if API is offline

### Vanilla JS fetch (any framework)

```javascript
const result = await fetch("http://localhost:5000/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: userInput })
}).then(r => r.json());

console.log(result.result.level_name);     // "Moderate Stress"
console.log(result.result.intervention);   // "Wellness workshop recommended"
```

---

## Docker Deployment

```bash
docker build -t mindcare-api .
docker run -p 5000:5000 -v $(pwd)/models:/app/models mindcare-api
```

---

## Multi-Modal Fusion Formula

```
Final_Score = 0.40 × Social_Score + 0.35 × Assessment_Score + 0.25 × Activity_Score
```

Adaptive weights when a source is unavailable:
- No social media → 0% / 58% / 42%  
- No activity data → 50% / 20% / 30%

---

## 8-Level Classification Schema

| Level | Name                     | Intervention                         |
|-------|--------------------------|--------------------------------------|
| 0     | Normal                   | None required                        |
| 1     | Mild Stress              | Self-help resources                  |
| 2     | Moderate Stress          | Wellness workshop                    |
| 3     | Severe Stress            | Counselling referral                 |
| 4     | Mild Anxiety             | Relaxation + counselling             |
| 5     | Moderate Anxiety         | Immediate counselling + CBT          |
| 6     | Suicidal Ideation        | Urgent counselling referral          |
| 7     | Depression Risk (Urgent) | ⚠️ Crisis intervention               |

---

## Target Performance (MindCare SRS)

| Metric    | Target |
|-----------|--------|
| Accuracy  | > 90%  |
| Precision | > 88%  |
| Recall    | > 85%  |
| F1-Score  | > 87%  |
| Kappa     | > 0.75 |
