"""
Global configuration parameters for the Market Microstructure DL Pipeline.
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"

# Deep Learning Hyperparameters
SEQUENCE_LENGTH = 60      # Lookback window (e.g., 60 minutes)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3

# Ensure core directories exist upon import
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_WEIGHTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
