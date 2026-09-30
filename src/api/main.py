from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.autoencoder import MarketAutoencoder
from models.lstm import LSTMPredictor
import config

app = FastAPI(
    title="Market Microstructure API",
    description="Clean backend serving deep learning market models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cpu'
autoencoder = None
lstm_model = None

# Global state to hold the latest live prediction
latest_market_state = {
    "regime": 0,          
    "prediction": 0.0,
    "timestamp": 0
}

@app.on_event("startup")
def load_models():
    global autoencoder, lstm_model
    try:
        N_FEATURES = 5
        
        autoencoder = MarketAutoencoder(seq_len=config.SEQUENCE_LENGTH, n_features=N_FEATURES)
        ae_path = config.MODEL_WEIGHTS_DIR / "autoencoder.pth"
        if ae_path.exists():
            autoencoder.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
            autoencoder.eval()
        
        lstm_model = LSTMPredictor(n_features=N_FEATURES)
        lstm_path = config.MODEL_WEIGHTS_DIR / "lstm.pth"
        if lstm_path.exists():
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=True))
            lstm_model.eval()
            
        print("Models loaded successfully into memory for inference!")
    except Exception as e:
        print(f"Warning: Could not load model weights. {e}")

class MarketDataRequest(BaseModel):
    sequence: list[list[float]]

@app.get("/health")
def health_check():
    return {"status": "online", "models_loaded": autoencoder is not None}

@app.get("/predict/latest")
def get_latest_prediction():
    """Endpoint for the frontend to poll the latest live data state."""
    return latest_market_state

@app.post("/predict/regime")
def predict_regime(data: MarketDataRequest):
    if not autoencoder:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        seq_tensor = torch.tensor([data.sequence], dtype=torch.float32).to(device)
        with torch.no_grad():
            _, latent = autoencoder(seq_tensor)
            
        # Update global state (Mapping latent to a pseudo-cluster 0-3 for UI)
        # In production, we'd use the saved KMeans model here.
        val = sum(latent[0].tolist())
        latest_market_state["regime"] = int(abs(val) % 4) 
        latest_market_state["timestamp"] = time.time()
        
        return {"status": "success", "latent_vector": latent[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/return")
def predict_return(data: MarketDataRequest):
    if not lstm_model:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        seq_tensor = torch.tensor([data.sequence], dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction = lstm_model(seq_tensor)
            
        pred_val = float(prediction[0][0].item())
        latest_market_state["prediction"] = pred_val
        latest_market_state["timestamp"] = time.time()
        
        return {"status": "success", "predicted_return": pred_val}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
