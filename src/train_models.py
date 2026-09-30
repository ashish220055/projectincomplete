import sys
import torch
from pathlib import Path

# Add src to path so we can import easily if run from root
sys.path.append(str(Path(__file__).resolve().parent))

import config
from data_pipeline import load_and_clean_data, generate_features, scale_data
from dataset import create_dataloaders
from models.autoencoder import MarketAutoencoder
from models.lstm import LSTMPredictor
from trainer import Trainer
from evaluation import extract_latent_embeddings, cluster_regimes, evaluate_lstm

def main():
    print("--- 1. Data Pipeline ---")
    raw_data_path = config.RAW_DATA_DIR / "BTCUSDT_1m_sample.csv"
    if not raw_data_path.exists():
        print(f"Error: Data not found at {raw_data_path}. Run download_data.py first.")
        sys.exit(1)
        
    df = load_and_clean_data(raw_data_path)
    features_df = generate_features(df)
    
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]
    
    train_scaled, test_scaled, _ = scale_data(train_df, test_df)
    
    train_loader, test_loader = create_dataloaders(
        train_scaled, test_scaled, config.SEQUENCE_LENGTH, config.BATCH_SIZE
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n--- 2. Training Autoencoder ---")
    n_features = train_scaled.shape[1]
    autoencoder = MarketAutoencoder(seq_len=config.SEQUENCE_LENGTH, n_features=n_features)
    
    ae_trainer = Trainer(autoencoder, device=device)
    # Train for 5 epochs for the MVP demonstration
    ae_trainer.train_autoencoder(train_loader, test_loader, epochs=5, patience=2)
    
    print("\n--- 3. Evaluating Autoencoder & Regimes ---")
    # Load best weights
    autoencoder.load_state_dict(torch.load(config.MODEL_WEIGHTS_DIR / "autoencoder.pth"))
    embeddings = extract_latent_embeddings(autoencoder, test_loader, device)
    labels, kmeans_model, sil_score = cluster_regimes(embeddings, n_clusters=4)
    print(f"Discovered 4 Market Regimes. Silhouette Score: {sil_score:.4f}")
    
    print("\n--- 4. Training LSTM Predictor ---")
    lstm_model = LSTMPredictor(n_features=n_features)
    lstm_trainer = Trainer(lstm_model, device=device)
    lstm_trainer.train_lstm(train_loader, test_loader, epochs=5, patience=2)
    
    print("\n--- 5. Evaluating LSTM ---")
    lstm_model.load_state_dict(torch.load(config.MODEL_WEIGHTS_DIR / "lstm.pth"))
    mse, acc = evaluate_lstm(lstm_model, test_loader, device)
    print(f"LSTM Test MSE: {mse:.6f}")
    print(f"LSTM Directional Accuracy: {acc * 100:.2f}%")
    
    print("\nPipeline Complete! Models are saved in models/weights/")

if __name__ == "__main__":
    main()
