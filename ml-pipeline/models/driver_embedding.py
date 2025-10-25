import torch
import torch.nn as nn

class DriverEmbedding(nn.Module):
    """
    Driver style embedding model
    Learns personalized driver representations from clean laps
    sequence2vec: Transformer CLS token → d_driver ∈ R^32
    """
    
    def __init__(
        self,
        input_dim=16,
        hidden_dim=128,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Embedding projection
        self.embedding_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # Multi-task heads for supervision
        self.sector_delta_head = nn.Linear(embedding_dim, 1)
        self.throttle_discipline_head = nn.Linear(embedding_dim, 1)
        self.brake_bias_head = nn.Linear(embedding_dim, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            embedding: (batch, embedding_dim)
            auxiliary_outputs: dict with multi-task predictions
        """
        
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Project input
        x = self.input_proj(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)
        
        # Take CLS token
        cls_output = x[:, 0, :]
        
        # Project to embedding
        embedding = self.embedding_proj(cls_output)
        
        # Auxiliary predictions for multi-task supervision
        auxiliary_outputs = {
            'sector_delta': self.sector_delta_head(embedding),
            'throttle_discipline': self.throttle_discipline_head(embedding),
            'brake_bias': self.brake_bias_head(embedding)
        }
        
        return embedding, auxiliary_outputs

if __name__ == "__main__":
    print("Testing Driver Embedding Model...")
    
    model = DriverEmbedding(input_dim=16, hidden_dim=128, embedding_dim=32)
    
    x = torch.randn(4, 200, 16)
    embedding, aux_outputs = model(x)
    
    print(f"✓ Model initialized")
    print(f"  Input: {x.shape}")
    print(f"  Embedding: {embedding.shape}")
    print(f"  Auxiliary outputs: {list(aux_outputs.keys())}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

