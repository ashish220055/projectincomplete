import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MarketSequenceDataset(Dataset):
    """
    PyTorch Dataset that creates sliding windows of market data.
    X: Shape (Sequence_Length, Features)
    y: Target return at the next timestep
    """
    def __init__(self, data: pd.DataFrame, sequence_length: int, target_col_idx: int = 0):
        self.data = data.values
        self.sequence_length = sequence_length
        self.target_col_idx = target_col_idx
        
    def __len__(self):
        # We need enough data for the sequence + 1 for the target
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        # The sequence window: (Sequence_Length, Features)
        x = self.data[idx : idx + self.sequence_length]
        
        # The target is the 'log_return' of the next step after the window
        y = self.data[idx + self.sequence_length, self.target_col_idx]
        
        # Add a dimension to y so it's shaped [1] instead of a scalar
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)

def create_dataloaders(train_scaled: pd.DataFrame, test_scaled: pd.DataFrame, 
                       seq_length: int, batch_size: int, target_col: str = 'log_return'):
    """Builds Train and Test PyTorch DataLoaders."""
    
    target_idx = train_scaled.columns.get_loc(target_col)
    
    train_dataset = MarketSequenceDataset(train_scaled, seq_length, target_idx)
    test_dataset = MarketSequenceDataset(test_scaled, seq_length, target_idx)
    
    # Shuffle train to prevent the model from learning the exact chronological path,
    # but sequences themselves are strictly chronological.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Don't shuffle test data so we can plot predictions chronologically later
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader
