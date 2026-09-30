import sys
from pathlib import Path
import config
from data_pipeline import load_and_clean_data, generate_features, scale_data
from dataset import create_dataloaders

def main():
    print("1. Loading raw data...")
    raw_data_path = config.RAW_DATA_DIR / "BTCUSDT_1m_sample.csv"
    
    if not raw_data_path.exists():
        print(f"Error: Could not find {raw_data_path}")
        sys.exit(1)
        
    df = load_and_clean_data(raw_data_path)
    print(f"Raw data shape: {df.shape}")
    
    print("\n2. Generating features...")
    features_df = generate_features(df)
    print(f"Features shape: {features_df.shape}")
    print("Features extracted:", list(features_df.columns))
    
    print("\n3. Splitting and scaling data (Train: 80%, Test: 20%)...")
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]
    
    train_scaled, test_scaled, scaler = scale_data(train_df, test_df)
    print(f"Train scaled shape: {train_scaled.shape}")
    print(f"Test scaled shape: {test_scaled.shape}")
    
    print("\n4. Creating PyTorch DataLoaders...")
    train_loader, test_loader = create_dataloaders(
        train_scaled, 
        test_scaled, 
        seq_length=config.SEQUENCE_LENGTH, 
        batch_size=config.BATCH_SIZE
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Fetch one batch to verify shapes
    x_batch, y_batch = next(iter(train_loader))
    print("\nSUCCESS! Pipeline verified.")
    print(f"Input Tensor Shape (Batch, Seq_Len, Features): {x_batch.shape}")
    print(f"Target Tensor Shape (Batch, 1): {y_batch.shape}")

if __name__ == "__main__":
    main()
