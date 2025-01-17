import torch
import torch.nn as nn

# Linearly scaling the dimensions of all the nodes to match a common dimension
# This step is done at the time of ingestion of data into the graph to maintain constant dimensionality

class FeatureProjector(nn.Module):
    def __init__(self, input_dims, common_dim):
        super().__init__()
        self.projectors = nn.ModuleDict({
            'transaction': nn.Linear(input_dims['transaction'], common_dim),
            'email': nn.Linear(input_dims['email'], common_dim),
            'phone': nn.Linear(input_dims['phone'], common_dim),
            'device': nn.Linear(input_dims['device'], common_dim)
            #### additional field not mentioned for data integrity 
        })

    def forward(self, node_type, features):
        return self.projectors[node_type](features)

input_dims = {
    'transaction': 32,  # example dimensions
    'email': 32,
    'phone': 32,
    'device': 64
    #### additional field not mentioned for data integrity 
}
common_dim = 128  # target common dimension for all node types
