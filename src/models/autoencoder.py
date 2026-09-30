import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int, embedding_dim: int = 64, latent_dim: int = 16):
        """
        Compresses the time-series window into a latent representation.
        """
        super(LSTMEncoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        
        # LSTM layer (batch_first=True expects (Batch, Seq_Len, Features))
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Compress to latent space (Market Regime)
        self.fc = nn.Linear(embedding_dim, latent_dim)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        x, (hidden, cell) = self.rnn(x)
        
        # Take the last hidden state of the sequence
        # x[:, -1, :] shape: (Batch, embedding_dim)
        last_hidden_state = x[:, -1, :]
        
        # Compress to latent space
        latent = self.fc(last_hidden_state)  # (Batch, latent_dim)
        return latent

class LSTMDecoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int, embedding_dim: int = 64, latent_dim: int = 16):
        """
        Reconstructs the original time-series from the latent representation.
        """
        super(LSTMDecoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        
        # Expand latent back to embedding dim
        self.fc = nn.Linear(latent_dim, embedding_dim)
        
        # LSTM layer
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Map back to original features
        self.output_layer = nn.Linear(embedding_dim, n_features)
        
    def forward(self, x):
        # x shape: (Batch, latent_dim)
        x = self.fc(x)  # (Batch, embedding_dim)
        
        # Repeat the latent vector seq_len times to feed into LSTM
        # shape: (Batch, Seq_Len, embedding_dim)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # LSTM
        x, (hidden, cell) = self.rnn(x)
        
        # Output mapping
        # shape: (Batch, Seq_Len, n_features)
        reconstruction = self.output_layer(x)
        return reconstruction

class MarketAutoencoder(nn.Module):
    """
    End-to-end Autoencoder for Pattern Discovery.
    Finds structural similarities in market conditions by compressing
    the microstructure features into a Latent Space and rebuilding them.
    """
    def __init__(self, seq_len: int, n_features: int, embedding_dim: int = 64, latent_dim: int = 16):
        super(MarketAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(seq_len, n_features, embedding_dim, latent_dim)
        self.decoder = LSTMDecoder(seq_len, n_features, embedding_dim, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Features)
        Returns:
            reconstruction: Tensor of shape (Batch, Seq_Len, Features)
            latent: Tensor of shape (Batch, Latent_Dim)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
