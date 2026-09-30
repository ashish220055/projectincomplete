from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import uvicorn

# Initialize FastAPI App
app = FastAPI(
    title="Market Microstructure API",
    description="Clean backend to serve deep learning market models",
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

# Define request schemas
class MarketDataRequest(BaseModel):
    # Expecting a flattened list of 60 periods * 5 features
    sequence: list[list[float]]

@app.get("/health")
def health_check():
    """Check if the backend is running."""
    return {"status": "online", "model_versions": {"autoencoder": "v1", "lstm": "v1"}}

@app.post("/predict/regime")
def predict_regime(data: MarketDataRequest):
    """
    Takes a sequence of market data, passes it through the Autoencoder,
    and returns the latent embedding (Market Regime fingerprint).
    """
    # In production, we would load the trained .pth model here
    # model.load_state_dict(...)
    
    # Placeholder response assuming 16-dim latent space
    return {
        "status": "success", 
        "latent_vector": [0.0] * 16,
        "message": "Model weights not yet connected in this MVP skeleton."
    }

@app.post("/predict/return")
def predict_return(data: MarketDataRequest):
    """
    Takes a sequence of market data, passes it through the LSTM,
    and returns the predicted next return.
    """
    # Placeholder response
    return {
        "status": "success", 
        "predicted_return": 0.0015,
        "message": "Model weights not yet connected in this MVP skeleton."
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
