from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

# -------------------------
# Load model and label encoder
# -------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)  # Full pipeline (TF-IDF + NB)

with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="News Category Classifier API")

class TextItem(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(request: TextItem):
    try:
        # Predict numeric label
        pred_numeric = model.predict([request.text])
        
        # Convert numeric label to readable category
        pred_category = le.inverse_transform(pred_numeric)
        
        return {
            "text": request.text,
            "predicted_category": pred_category[0]
        }
    
    except Exception as e:
        return {"error": str(e)}
