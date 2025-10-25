import torch
import torch.nn as nn

class AnomalyDetector(nn.Module):
    """
    LSTM Autoencoder for anomaly detection in telemetry
    Detects unusual patterns in sensor readings
    """
    
    def __init__(
        self,
        input_dim=16,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            reconstructed: (batch, seq_len, input_dim)
            encoding: (batch, hidden_dim)
        """
        
        # Encode
        encoded, (h_n, c_n) = self.encoder(x)
        
        # Take last hidden state as encoding
        encoding = h_n[-1]  # (batch, hidden_dim)
        
        # Decode
        # Repeat encoding for each timestep
        seq_len = x.size(1)
        decoder_input = encoding.unsqueeze(1).repeat(1, seq_len, 1)
        
        decoded, _ = self.decoder(decoder_input, (h_n, c_n))
        
        # Project back to input space
        reconstructed = self.output_proj(decoded)
        
        return reconstructed, encoding
    
    def compute_anomaly_score(self, x):
        """Compute reconstruction error as anomaly score"""
        reconstructed, _ = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return mse

if __name__ == "__main__":
    print("Testing Anomaly Detector...")
    
    model = AnomalyDetector(input_dim=16, hidden_dim=64, num_layers=2)
    
    x = torch.randn(4, 100, 16)
    reconstructed, encoding = model(x)
    anomaly_scores = model.compute_anomaly_score(x)
    
    print(f"✓ Model initialized")
    print(f"  Input: {x.shape}")
    print(f"  Reconstructed: {reconstructed.shape}")
    print(f"  Encoding: {encoding.shape}")
    print(f"  Anomaly scores: {anomaly_scores.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

