# Deep Learning Market Microstructure Platform

This project aims to discover latent market regimes using Autoencoders and predict baseline price movements using LSTMs on cryptocurrency data.

## Architecture
Please refer to the [ARCHITECTURE.md](ARCHITECTURE.md) file for the complete blueprint, interface contracts, and module breakdown.

## Setup Instructions
1. Install Python 3.9+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place raw minute-level cryptocurrency CSV data (e.g., BTC/USDT) into `data/raw/`.
4. Run the data pipeline and training scripts located in `src/` (details to follow as development progresses).
