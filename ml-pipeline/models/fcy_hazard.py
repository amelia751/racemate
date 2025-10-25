import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from tire_degradation import TemporalConvNet

class FCYHazardModel(nn.Module):
    """
    Full-course yellow (caution) probability predictor
    TCN-based hazard model with survival analysis
    """
    
    def __init__(
        self,
        input_dim=16,
        hidden_channels=128,
        kernel_size=3,
        num_layers=3,
        horizon_laps=6
    ):
        super().__init__()
        
        self.horizon_laps = horizon_laps
        
        # TCN backbone
        self.tcn = TemporalConvNet(
            input_dim,
            [hidden_channels] * num_layers,
            kernel_size=kernel_size
        )
        
        # Hazard rate head (per lap in horizon)
        self.hazard_head = nn.Linear(hidden_channels, horizon_laps)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            hazard_rates: (batch, horizon_laps) - prob of FCY in each lap
            cumulative_prob: (batch, 1) - prob of FCY within horizon
        """
        
        # TCN
        x = x.transpose(1, 2)  # (B, C, L)
        features = self.tcn(x)
        features = features[:, :, -1]  # Last timestep
        
        # Hazard rates
        hazard_logits = self.hazard_head(features)
        hazard_rates = self.sigmoid(hazard_logits)
        
        # Cumulative probability (1 - product of survival)
        survival_probs = 1 - hazard_rates
        cumulative_survival = torch.prod(survival_probs, dim=1, keepdim=True)
        cumulative_prob = 1 - cumulative_survival
        
        return hazard_rates, cumulative_prob

if __name__ == "__main__":
    print("Testing FCY Hazard Model...")
    
    model = FCYHazardModel(input_dim=16, hidden_channels=128, num_layers=3)
    
    x = torch.randn(4, 200, 16)
    hazard_rates, cumulative_prob = model(x)
    
    print(f"✓ Model initialized")
    print(f"  Hazard rates: {hazard_rates.shape}")
    print(f"  Cumulative prob: {cumulative_prob.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

