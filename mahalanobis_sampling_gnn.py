import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import numpy as np

class HeteroGCN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, num_classes, feature_mean, feature_cov):
        super(HeteroGCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h1_feats)
        self.conv2 = GraphConv(h1_feats, h2_feats)
        self.conv3 = GraphConv(h2_feats, num_classes)
        
        self.mean = feature_mean
        self.cov = feature_cov
        self.inv_cov = np.linalg.inv(self.cov)

    def mahalanobis(self, x, y):
        delta = x - y
        distance = np.sqrt(np.dot(np.dot(delta, self.inv_cov), delta.T))
        return distance

    def forward(self, graph, inputs):
        h = inputs
        for layer in [self.conv1, self.conv2, self.conv3]:
            new_h = torch.zeros_like(h)
            for node in range(graph.number_of_nodes()):
                neighbors = graph.neighbors(node)
                neighbor_feats = h[neighbors]
                
                # Compute distances and select top 10
                distances = [self.mahalanobis(inputs[node].detach().numpy(), feat.detach().numpy()) for feat in neighbor_feats]
                top_neighbors = neighbors[np.argsort(distances)[:10]]

                # Aggregate only top neighbors' features
                agg_feat = torch.mean(h[top_neighbors], dim=0)
                new_h[node] = layer(graph, agg_feat)
                
            h = F.relu(new_h)
        return h

# Global node features
node_features = global.nodes

# Calculate the mean of the features
feature_mean = np.mean(node_features, axis=0)

# Calculate the covariance matrix of the features
feature_cov = np.cov(node_features, rowvar=False)  # rowvar=False indicates that rows represent variables

print("Feature Mean:", feature_mean)
print("Feature Covariance Matrix:\n", feature_cov)
