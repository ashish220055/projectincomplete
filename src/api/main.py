from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import sys
from pathlib import Path

# Add src to path so we can import our models
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.autoencoder import MarketAutoencoder
from models.lstm import LSTMPredictor
import config

# Initialize FastAPI App
app = FastAPI(
    title="Market Microstructure API",
    description="Clean backend serving deep learning market models",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models
device = 'cpu' # API inference is fine on CPU
autoencoder = None
lstm_model = None

@app.on_event("startup")
def load_models():
    """Loads the trained .pth weights into memory when the server starts."""
    global autoencoder, lstm_model
    try:
        # In our pipeline we engineered 5 features
        N_FEATURES = 5
        
        # Load Autoencoder
        autoencoder = MarketAutoencoder(seq_len=config.SEQUENCE_LENGTH, n_features=N_FEATURES)
        ae_path = config.MODEL_WEIGHTS_DIR / "autoencoder.pth"
        if ae_path.exists():
            # weights_only=True for security (prevents arbitrary code execution in pickles)
            autoencoder.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
            autoencoder.eval()
        
        # Load LSTM
        lstm_model = LSTMPredictor(n_features=N_FEATURES)
        lstm_path = config.MODEL_WEIGHTS_DIR / "lstm.pth"
        if lstm_path.exists():
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=True))
            lstm_model.eval()
            
        print("Models loaded successfully into memory for inference!")
    except Exception as e:
        print(f"Warning: Could not load model weights. {e}")

class MarketDataRequest(BaseModel):
    # Expecting a sequence of 60 periods * 5 features
    sequence: list[list[float]]

@app.get("/health")
def health_check():
    return {"status": "online", "models_loaded": autoencoder is not None}

@app.post("/predict/regime")
def predict_regime(data: MarketDataRequest):
    """Passes the incoming sequence through the Autoencoder to extract latent state."""
    if not autoencoder:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    try:
        # Convert list to tensor: Shape (Batch=1, Seq=60, Features=5)
        seq_tensor = torch.tensor([data.sequence], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            _, latent = autoencoder(seq_tensor)
            
        return {
            "status": "success", 
            "latent_vector": latent[0].tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/return")
def predict_return(data: MarketDataRequest):
    """Passes the incoming sequence through the LSTM to predict the next return."""
    if not lstm_model:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    try:
        seq_tensor = torch.tensor([data.sequence], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            prediction = lstm_model(seq_tensor)
            
        return {
            "status": "success", 
            "predicted_return": float(prediction[0][0].item()),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
