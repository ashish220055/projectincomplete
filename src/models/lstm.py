import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    """
    Supervised LSTM for Baseline Price Prediction.
    Takes a window of market microstructure features and attempts 
    to predict the target return at the next timestep.
    """
    def __init__(self, n_features: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Core LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers for prediction
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Final output layer (predicts a single continuous return value)
        self.out = nn.Linear(32, 1)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Features)
        Returns:
            prediction: Tensor of shape (Batch, 1)
        """
        # Pass sequence through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # We only care about the last time step's output to make a forward prediction
        # lstm_out[:, -1, :] shape: (Batch, hidden_dim)
        last_step_out = lstm_out[:, -1, :]
        
        # Pass through dense layers
        x = self.fc1(last_step_out)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Final prediction
        prediction = self.out(x)
        return prediction
