import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import FraudYelpDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import random

# Load the dataset
dataset = FraudYelpDataset()
graph = dataset[0]

# Extract node features and labels
features = graph.ndata['feature']
labels = graph.ndata['label']

# Train-test split
train_idx, test_idx = train_test_split(
    torch.arange(features.shape[0]),
    test_size=0.2,
    stratify=labels,
    random_state=42,
)
val_idx, test_idx = train_test_split(
    test_idx,
    test_size=0.5,
    stratify=labels[test_idx],
    random_state=42,
)

# Compute class weights
num_fraud = (labels == 1).sum().item()
num_non_fraud = (labels == 0).sum().item()
total = num_fraud + num_non_fraud
class_weights = torch.tensor([total / num_non_fraud, total / num_fraud], dtype=torch.float32)

def normalize_edge_weights(weights):
    """Normalize edge weights to range [0, 1]."""
    min_val, max_val = weights.min(), weights.max()
    if max_val > min_val:
        return (weights - min_val) / (max_val - min_val)
    else:
        return torch.zeros_like(weights)

# Assign edge weights
def assign_rur_weights(graph):
    """Assign weights to R-U-R edges."""
    labels = graph.ndata['label']
    features = graph.ndata['feature']
    src, dst = graph.edges(etype='net_rur')
    edge_weights = torch.zeros(graph.num_edges('net_rur'))

    for i in range(len(src)):
        if labels[src[i]] == 1 and labels[dst[i]] == 1:
            label_score = 1.0
        elif labels[src[i]] == 1 or labels[dst[i]] == 1:
            label_score = 0.9
        else:
            label_score = 0.0

        src_feat = features[src[i]]
        dst_feat = features[dst[i]]
        cos_sim = F.cosine_similarity(src_feat.unsqueeze(0), dst_feat.unsqueeze(0)).item()
        cos_sim = (cos_sim + 1) / 2
        edge_weights[i] = label_score + cos_sim

    neighbor_edge_scores = torch.zeros_like(edge_weights)
    for i in range(len(src)):
        src_neighbors = graph.out_edges(src[i], etype='net_rur')[1]
        dst_neighbors = graph.out_edges(dst[i], etype='net_rur')[1]
        for neighbor in src_neighbors:
            neighbor_edge_scores[i] += 0.25 * edge_weights[neighbor]
        for neighbor in dst_neighbors:
            neighbor_edge_scores[i] += 0.25 * edge_weights[neighbor]

    final_edge_weights = edge_weights + neighbor_edge_scores
    graph.edges['net_rur'].data['weight'] = normalize_edge_weights(final_edge_weights)

def assign_rsr_weights(graph):
    """Assign weights to R-S-R edges."""
    features = graph.ndata['feature']
    src, dst = graph.edges(etype='net_rsr')
    edge_weights = torch.zeros(graph.num_edges('net_rsr'))

    for i in range(len(src)):
        src_feat = features[src[i]]
        dst_feat = features[dst[i]]
        cos_sim = F.cosine_similarity(src_feat.unsqueeze(0), dst_feat.unsqueeze(0)).item()
        edge_weights[i] = (cos_sim + 1) / 2

    graph.edges['net_rsr'].data['weight'] = normalize_edge_weights(edge_weights)

def assign_rtr_weights(graph):
    """Assign weights to R-T-R edges."""
    features = graph.ndata['feature']
    src, dst = graph.edges(etype='net_rtr')
    edge_weights = torch.zeros(graph.num_edges('net_rtr'))

    for i in range(len(src)):
        src_feat = features[src[i]]
        dst_feat = features[dst[i]]
        cos_sim = F.cosine_similarity(src_feat.unsqueeze(0), dst_feat.unsqueeze(0)).item()
        edge_weights[i] = (cos_sim + 1) / 2

    graph.edges['net_rtr'].data['weight'] = normalize_edge_weights(edge_weights)

# Apply weight assignment functions
assign_rur_weights(graph)
assign_rsr_weights(graph)
assign_rtr_weights(graph)

def create_model(in_feats, hidden_feats, num_classes, edge_types, num_layers, dropout_rate):
    class HeteroGCN(nn.Module):
        def __init__(self):
            super(HeteroGCN, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(dgl.nn.HeteroGraphConv({
                etype: dgl.nn.GraphConv(in_feats, hidden_feats, norm='right')
                for etype in edge_types
            }))
            for _ in range(num_layers - 1):
                self.layers.append(dgl.nn.HeteroGraphConv({
                    etype: dgl.nn.GraphConv(hidden_feats, hidden_feats, norm='right')
                    for etype in edge_types
                }))
            self.out_layer = dgl.nn.HeteroGraphConv({
                etype: dgl.nn.GraphConv(hidden_feats, num_classes, norm='right')
                for etype in edge_types
            })
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, g, x):
            # Wrap input features into a dictionary with the node type as key
            x = {'review': x}
            for layer in self.layers:
                x = layer(g, x)  # Pass the dictionary to HeteroGraphConv
                x = {k: F.relu(v) for k, v in x.items()}  # Apply activation
                x = {k: self.dropout(v) for k, v in x.items()}
            x = self.out_layer(g, x)
            return x['review']  # Only the 'review' node type is used for output

    return HeteroGCN()

# Define hyperparameter ranges
param_grid = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "hidden_feats": [8, 16, 32, 64],
    "num_layers": [2, 3],
    "dropout_rate": [0.2, 0.5],
    "weight_decay": [1e-6, 1e-4]
}

# Evaluate a model with specific hyperparameters
def evaluate_model(params):
    learning_rate = params["learning_rate"]
    hidden_feats = params["hidden_feats"]
    num_layers = params["num_layers"]
    dropout_rate = params["dropout_rate"]
    weight_decay = params["weight_decay"]

    model = create_model(features.shape[1], hidden_feats, 2, graph.etypes, num_layers, dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        logits = model(graph, features)
        loss = criterion(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(graph, features)
        val_preds = torch.argmax(val_logits[val_idx], dim=1)
        val_f1 = classification_report(labels[val_idx].numpy(), val_preds.numpy(), output_dict=True)['macro avg']['f1-score']
    return val_f1

# Random search for hyperparameter tuning
num_trials = 10
best_params = None
best_f1 = 0.0

with tqdm(total=num_trials, desc="Hyperparameter Tuning", unit="trial") as pbar:
    for _ in range(num_trials):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        print(f"Testing params: {params}")
        val_f1 = evaluate_model(params)
        print(f"Validation F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params
        pbar.update(1)

print(f"\nBest Params: {best_params}")
print(f"Best Validation F1: {best_f1:.4f}")