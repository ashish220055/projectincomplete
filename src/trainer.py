import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import config

class Trainer:
    """
    Handles the training loops, loss calculations, and early stopping
    for both the Autoencoder and the LSTM Predictor.
    """
    def __init__(self, model: nn.Module, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        
    def train_autoencoder(self, train_loader, test_loader, epochs: int, patience: int = 5, save_path: str = "autoencoder.pth"):
        """Unsupervised training loop using MSE Loss for reconstruction."""
        criterion = nn.MSELoss()
        return self._run_training_loop(train_loader, test_loader, criterion, epochs, patience, save_path, is_autoencoder=True)

    def train_lstm(self, train_loader, test_loader, epochs: int, patience: int = 5, save_path: str = "lstm.pth"):
        """Supervised training loop using MSE Loss for next-step prediction."""
        criterion = nn.MSELoss()
        return self._run_training_loop(train_loader, test_loader, criterion, epochs, patience, save_path, is_autoencoder=False)

    def _run_training_loop(self, train_loader, test_loader, criterion, epochs, patience, save_name, is_autoencoder):
        best_val_loss = float('inf')
        patience_counter = 0
        
        save_path = config.MODEL_WEIGHTS_DIR / save_name
        
        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                if is_autoencoder:
                    # Autoencoder targets itself
                    reconstruction, _ = self.model(X_batch)
                    loss = criterion(reconstruction, X_batch)
                else:
                    # LSTM targets the y_batch
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation Phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    if is_autoencoder:
                        reconstruction, _ = self.model(X_batch)
                        loss = criterion(reconstruction, X_batch)
                    else:
                        predictions = self.model(X_batch)
                        loss = criterion(predictions, y_batch)
                        
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(test_loader)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> Best model saved to {save_path.name}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break
                    
        return best_val_loss
