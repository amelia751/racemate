import torch
import torch.nn as nn

class TireDegradationModel(nn.Module):
    """
    Physics-informed tire wear model
    Base: linear physics model
    Residual: TCN to learn corrections
    """
    
    def __init__(
        self,
        input_dim=16,
        hidden_channels=64,
        kernel_size=3,
        num_layers=3
    ):
        super().__init__()
        
        # Physics-based coefficients (learnable)
        self.alpha_brake = nn.Parameter(torch.tensor(0.001))  # brake energy coef
        self.beta_lateral = nn.Parameter(torch.tensor(0.001))  # lateral load coef
        self.gamma_temp = nn.Parameter(torch.tensor(0.01))  # temp coef
        
        # Residual TCN
        self.tcn = TemporalConvNet(
            input_dim,
            [hidden_channels] * num_layers,
            kernel_size=kernel_size,
            dropout=0.1
        )
        
        # Residual head
        self.residual_head = nn.Linear(hidden_channels, 1)
    
    def physics_model(self, features):
        """
        Physics-based grip loss
        features: dict with 'cum_brake_energy', 'cum_lateral_load', 'air_temp'
        """
        
        brake_wear = self.alpha_brake * features['cum_brake_energy']
        lateral_wear = self.beta_lateral * features['cum_lateral_load']
        temp_effect = self.gamma_temp * features.get('air_temp', torch.zeros_like(brake_wear))
        
        # Base grip loss
        grip_loss_base = brake_wear + lateral_wear + temp_effect
        
        return grip_loss_base
    
    def forward(self, x, features_dict):
        """
        Args:
            x: (batch, seq_len, input_dim) telemetry sequence
            features_dict: physics features
        Returns:
            grip_index: (batch, 1) predicted grip
        """
        
        # Physics base
        grip_loss_base = self.physics_model(features_dict)
        
        # Residual correction via TCN
        residual = self.tcn(x.transpose(1, 2))  # TCN expects (B, C, L)
        residual = residual[:, :, -1]  # Last timestep
        residual = self.residual_head(residual)
        
        # Combined
        total_grip_loss = grip_loss_base + residual
        
        # Grip index = 1 - loss (clamped)
        grip_index = torch.clamp(1.0 - total_grip_loss, 0.5, 1.0)
        
        return grip_index


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network"""
    
    def __init__(self, input_dim, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        
        layers = []
        num_levels = len(channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else channels[i-1]
            out_channels = channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    """Single TCN block with dilated convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2  # Causal padding
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual
        res = x if self.downsample is None else self.downsample(x)
        
        # Match dimensions if needed
        if out.size(2) != res.size(2):
            diff = out.size(2) - res.size(2)
            if diff > 0:
                out = out[:, :, :-diff]
            else:
                res = res[:, :, :diff]
        
        return self.relu(out + res)

if __name__ == "__main__":
    # Test model
    print("Testing Tire Degradation Model...")
    
    model = TireDegradationModel(input_dim=16, hidden_channels=64, num_layers=3)
    
    # Create dummy input
    batch_size = 4
    seq_len = 200
    input_dim = 16
    
    x = torch.randn(batch_size, seq_len, input_dim)
    features_dict = {
        'cum_brake_energy': torch.randn(batch_size, 1),
        'cum_lateral_load': torch.randn(batch_size, 1),
        'air_temp': torch.randn(batch_size, 1)
    }
    
    grip = model(x, features_dict)
    
    print(f"✓ Model initialized")
    print(f"  Input shape: {x.shape}")
    print(f"  Grip output: {grip.shape}")
    print(f"  Grip range: [{grip.min().item():.3f}, {grip.max().item():.3f}]")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

