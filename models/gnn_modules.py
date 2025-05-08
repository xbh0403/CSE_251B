import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class EdgeConv(MessagePassing):
    """Edge convolution layer for graph neural networks"""
    
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max')  # Max aggregation
        
        # MLP for processing node features together with edge features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 3, out_channels),  # *2 for concatenation, +3 for edge features
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr):
        # Add self-loops to edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Add zeros for edge attributes of self-loops
        self_loop_attr = torch.zeros((x.size(0), 3), dtype=edge_attr.dtype, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # Construct message using source node, target node, and edge features
        return self.mlp(torch.cat([x_i, x_j, edge_attr], dim=1))

class GNNEncoder(nn.Module):
    """Graph Neural Network for encoding agent interactions"""
    
    def __init__(self, node_dim, hidden_dim, num_layers=3):
        super(GNNEncoder, self).__init__()
        
        # Initial node feature transformation
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # Stack of EdgeConv layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(EdgeConv(hidden_dim, hidden_dim))
        
        # Final node feature transformation
        self.node_decoder = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        # Initial encoding
        x = self.node_encoder(x)
        x = F.relu(x)
        
        # Apply EdgeConv layers
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Final transformation
        x = self.node_decoder(x)
        
        return x