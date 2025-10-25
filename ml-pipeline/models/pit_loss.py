import torch
import torch.nn as nn

class PitLossModel(nn.Module):
    """
    Pit lane loss prediction model
    Estimates time loss from pit stop including merge penalty
    """
    
    def __init__(self, input_dim=16, hidden_dim=64):
        super().__init__()
        
        # Base pit time components
        self.pit_lane_speed_limit = nn.Parameter(torch.tensor(60.0))  # km/h
        self.pit_lane_length = nn.Parameter(torch.tensor(300.0))  # meters
        self.service_time_base = nn.Parameter(torch.tensor(12.0))  # seconds
        
        # Traffic merge penalty network
        self.merge_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # merge penalty in seconds
        )
    
    def forward(self, traffic_state):
        """
        Args:
            traffic_state: (batch, input_dim) traffic conditions near pit exit
        Returns:
            total_pit_loss: (batch, 1) total time loss in seconds
        """
        
        # Base pit time (lane + service)
        pit_lane_time = (self.pit_lane_length / 1000.0) / (self.pit_lane_speed_limit / 3600.0)
        base_pit_time = pit_lane_time + self.service_time_base
        
        # Merge penalty from traffic
        merge_penalty = self.merge_net(traffic_state)
        merge_penalty = torch.relu(merge_penalty)  # Non-negative
        
        # Total pit loss
        total_pit_loss = base_pit_time + merge_penalty
        
        return total_pit_loss

if __name__ == "__main__":
    print("Testing Pit Loss Model...")
    
    model = PitLossModel(input_dim=16, hidden_dim=64)
    
    traffic_state = torch.randn(4, 16)
    pit_loss = model(traffic_state)
    
    print(f"✓ Model initialized")
    print(f"  Pit loss output: {pit_loss.shape}")
    print(f"  Sample pit times: {pit_loss[:3].squeeze().tolist()}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

