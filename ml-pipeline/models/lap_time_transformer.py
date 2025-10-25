import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

class LapTimeTransformer(nn.Module):
    """
    Transformer for lap-time delta prediction
    Per spec: 4-layer, 256 hidden, 4 heads
    Inputs: 10s × 20Hz × ~16 signals = 200×16 tokens
    Output: Next lap time delta + quantiles
    """
    
    def __init__(
        self,
        input_dim=16,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        max_seq_len=200,
        quantiles=[0.1, 0.5, 0.9]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.quantiles = quantiles
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # Output heads
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in quantiles
        ])
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            mean: (batch, 1)
            quantiles: list of (batch, 1)
        """
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Take last token (or pool)
        x = x[:, -1, :]  # (batch, hidden_dim)
        
        # Predictions
        mean = self.mean_head(x)
        quantiles = [q_head(x) for q_head in self.quantile_heads]
        
        return mean, quantiles


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding"""
        return x + self.pe[:, :x.size(1), :]


class QuantileLoss(nn.Module):
    """Quantile regression loss"""
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, preds, target):
        """
        Args:
            preds: list of (batch, 1) quantile predictions
            target: (batch, 1) ground truth
        """
        losses = []
        
        for tau, pred in zip(self.quantiles, preds):
            error = target - pred
            loss = torch.max(
                tau * error,
                (tau - 1) * error
            )
            losses.append(loss.mean())
        
        return sum(losses) / len(losses)

if __name__ == "__main__":
    # Test model
    print("Testing Lap Time Transformer...")
    
    model = LapTimeTransformer(input_dim=16, hidden_dim=256, num_layers=4)
    
    # Create dummy input
    batch_size = 4
    seq_len = 200
    input_dim = 16
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    mean, quantiles = model(x)
    
    print(f"✓ Model initialized")
    print(f"  Input shape: {x.shape}")
    print(f"  Mean output: {mean.shape}")
    print(f"  Quantile outputs: {[q.shape for q in quantiles]}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

