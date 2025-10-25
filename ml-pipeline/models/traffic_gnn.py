import torch
import torch.nn as nn
try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. Install with: pip install torch-geometric")

if TORCH_GEOMETRIC_AVAILABLE:
    class TrafficGNN(nn.Module):
        """
        Graph Neural Network for traffic interactions
        Nodes = cars, Edges = proximity < threshold
        Predicts: traffic loss (ms) and overtake probability
        """
        
        def __init__(
            self,
            node_feature_dim=32,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1
        ):
            super().__init__()
            
            # GraphSAGE layers
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(node_feature_dim, hidden_dim))
            
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            
            # Readout MLP
            self.traffic_loss_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # ms of time loss
            )
            
            self.overtake_prob_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),  # probability
                nn.Sigmoid()
            )
        
        def forward(self, x, edge_index, batch=None):
            """
            Args:
                x: (num_nodes, node_feature_dim) node features
                edge_index: (2, num_edges) graph connectivity
                batch: (num_nodes,) batch assignment
            Returns:
                traffic_loss: (num_graphs, 1)
                overtake_prob: (num_graphs, 1)
            """
            
            # GNN layers
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = self.relu(x)
                
                if i < len(self.convs) - 1:
                    x = self.dropout(x)
            
            # Global pooling per graph
            if batch is not None:
                x = global_mean_pool(x, batch)
            else:
                x = x.mean(dim=0, keepdim=True)
            
            # Predictions
            traffic_loss = self.traffic_loss_head(x)
            overtake_prob = self.overtake_prob_head(x)
            
            return traffic_loss, overtake_prob
else:
    # Simplified Traffic GNN using standard PyTorch (no torch-geometric needed)
    class TrafficGNN(nn.Module):
        """
        Simplified Graph Neural Network for traffic interactions
        Uses attention mechanism instead of torch-geometric
        Nodes = cars, predicts traffic loss and overtake probability
        """
        
        def __init__(
            self,
            node_feature_dim=32,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1
        ):
            super().__init__()
            
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            # Node feature projection
            self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
            
            # Attention-based aggregation layers
            self.attention_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.attention_layers.append(
                    nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
                )
            
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            
            # Readout MLP
            self.traffic_loss_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)  # ms of time loss
            )
            
            self.overtake_prob_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),  # probability
                nn.Sigmoid()
            )
        
        def forward(self, x, edge_index=None, batch=None):
            """
            Args:
                x: (batch, num_nodes, node_feature_dim) or (num_nodes, node_feature_dim)
                edge_index: (optional) not used in simplified version
                batch: (optional) not used in simplified version
            Returns:
                traffic_loss: (batch, 1) or (1, 1)
                overtake_prob: (batch, 1) or (1, 1)
            """
            
            # Handle both batched and unbatched input
            if x.dim() == 2:
                x = x.unsqueeze(0)  # (1, num_nodes, node_feature_dim)
            
            # Project node features
            x = self.node_proj(x)  # (batch, num_nodes, hidden_dim)
            
            # Apply attention layers (nodes attend to each other)
            for i in range(self.num_layers):
                # Self-attention among nodes
                attn_out, _ = self.attention_layers[i](x, x, x)
                x = self.layer_norms[i](x + attn_out)  # Residual connection
                
                if i < self.num_layers - 1:
                    x = self.dropout(self.relu(x))
            
            # Global pooling (mean over nodes)
            x = x.mean(dim=1)  # (batch, hidden_dim)
            
            # Predictions
            traffic_loss = self.traffic_loss_head(x)
            overtake_prob = self.overtake_prob_head(x)
            
            return traffic_loss, overtake_prob

if __name__ == "__main__":
    if TORCH_GEOMETRIC_AVAILABLE:
        print("Testing Traffic GNN...")
        
        model = TrafficGNN(node_feature_dim=32, hidden_dim=64, num_layers=2)
        
        # Dummy graph data
        num_nodes = 10
        node_features = torch.randn(num_nodes, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        
        traffic_loss, overtake_prob = model(node_features, edge_index)
        
        print(f"✓ Model initialized")
        print(f"  Nodes: {num_nodes}")
        print(f"  Traffic loss output: {traffic_loss.shape}")
        print(f"  Overtake prob output: {overtake_prob.shape}")
    else:
        print("✗ torch_geometric not available")

