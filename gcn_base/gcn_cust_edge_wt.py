import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dgl.data import FraudYelpDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create directory for saving edge weights
EDGE_WEIGHTS_DIR = "edge_weights"
os.makedirs(EDGE_WEIGHTS_DIR, exist_ok=True)


# Function to save and load edge weights
def save_edge_weights(edge_weights, filename):
    """Save edge weights to a file."""
    filepath = os.path.join(EDGE_WEIGHTS_DIR, filename)
    torch.save(edge_weights, filepath)


def load_edge_weights(filename):
    """Load edge weights from a file."""
    filepath = os.path.join(EDGE_WEIGHTS_DIR, filename)
    if os.path.exists(filepath):
        return torch.load(filepath)
    return None


def normalize_edge_weights(weights):
    """Normalize edge weights to range [0, 1]."""
    min_val, max_val = weights.min(), weights.max()
    if max_val > min_val:
        return (weights - min_val) / (max_val - min_val)
    else:
        return torch.zeros_like(weights)


def assign_rur_weights(graph):
    filename = "rur_weights.pt"
    edge_weights = load_edge_weights(filename)
    if edge_weights is not None:
        graph.edges['net_rur'].data['weight'] = edge_weights
        return

    labels = graph.ndata['label']
    features = graph.ndata['feature']
    src, dst = graph.edges(etype='net_rur')
    edge_weights = torch.zeros(graph.num_edges('net_rur'))

    for i in tqdm(range(len(src)), desc="Calculating R-U-R edge weights"):
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
    for i in tqdm(range(len(src)), desc="Incorporating neighbor influence for R-U-R edges"):
        src_neighbors = graph.out_edges(src[i], etype='net_rur')[1]
        dst_neighbors = graph.out_edges(dst[i], etype='net_rur')[1]

        for neighbor in src_neighbors:
            neighbor_edge_scores[i] += 0.25 * edge_weights[neighbor]
        for neighbor in dst_neighbors:
            neighbor_edge_scores[i] += 0.25 * edge_weights[neighbor]

    final_edge_weights = edge_weights + neighbor_edge_scores
    final_edge_weights = normalize_edge_weights(final_edge_weights)
    graph.edges['net_rur'].data['weight'] = final_edge_weights

    save_edge_weights(final_edge_weights, filename)


def assign_rsr_weights(graph):
    filename = "rsr_weights.pt"
    edge_weights = load_edge_weights(filename)
    if edge_weights is not None:
        graph.edges['net_rsr'].data['weight'] = edge_weights
        return

    features = graph.ndata['feature']
    src, dst = graph.edges(etype='net_rsr')
    edge_weights = torch.zeros(graph.num_edges('net_rsr'))

    for i in tqdm(range(len(src)), desc="Calculating R-S-R edge weights"):
        src_feat = features[src[i]]
        dst_feat = features[dst[i]]
        cos_sim = F.cosine_similarity(src_feat.unsqueeze(0), dst_feat.unsqueeze(0)).item()
        edge_weights[i] = (cos_sim + 1) / 2

    edge_weights = normalize_edge_weights(edge_weights)
    graph.edges['net_rsr'].data['weight'] = edge_weights
    save_edge_weights(edge_weights, filename)


def assign_rtr_weights(graph):
    filename = "rtr_weights.pt"
    edge_weights = load_edge_weights(filename)
    if edge_weights is not None:
        graph.edges['net_rtr'].data['weight'] = edge_weights
        return

    features = graph.ndata['feature']
    src, dst = graph.edges(etype='net_rtr')
    edge_weights = torch.zeros(graph.num_edges('net_rtr'))

    for i in tqdm(range(len(src)), desc="Calculating R-T-R edge weights"):
        src_feat = features[src[i]]
        dst_feat = features[dst[i]]
        cos_sim = F.cosine_similarity(src_feat.unsqueeze(0), dst_feat.unsqueeze(0)).item()
        edge_weights[i] = (cos_sim + 1) / 2

    edge_weights = normalize_edge_weights(edge_weights)
    graph.edges['net_rtr'].data['weight'] = edge_weights
    save_edge_weights(edge_weights, filename)


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

# Assign edge weights
assign_rur_weights(graph)
assign_rsr_weights(graph)
assign_rtr_weights(graph)

# Define a Heterogeneous GCN Model with Edge Weights
class HeteroGCNWithEdgeWeights(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, edge_types):
        super(HeteroGCNWithEdgeWeights, self).__init__()
        self.conv1 = dgl.nn.HeteroGraphConv({
            etype: dgl.nn.GraphConv(in_feats, hidden_feats, norm='right')
            for etype in edge_types
        })
        self.conv2 = dgl.nn.HeteroGraphConv({
            etype: dgl.nn.GraphConv(hidden_feats, num_classes, norm='right')
            for etype in edge_types
        })

    def forward(self, g, x):
        # Get edge weights for each type
        edge_weights = {etype: g.edges[etype].data['weight'] for etype in g.etypes}
        # Apply the first layer with edge weights
        x = self.conv1(g, {'review': x}, mod_kwargs={'edge_weight': edge_weights})
        x = {k: F.relu(v) for k, v in x.items()}  # Apply activation
        # Apply the second layer
        x = self.conv2(g, x, mod_kwargs={'edge_weight': edge_weights})
        # Only the 'review' node type is used for output
        return x['review']

# Initialize the model
in_feats = features.shape[1]
hidden_feats = 16
num_classes = 2  # Binary classification (fraud or non-fraud)
model = HeteroGCNWithEdgeWeights(in_feats, hidden_feats, num_classes, graph.etypes)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Compute class weights
num_fraud = (labels == 1).sum().item()
num_non_fraud = (labels == 0).sum().item()
total = num_fraud + num_non_fraud
class_weights = torch.tensor([total / num_non_fraud, total / num_fraud], dtype=torch.float32)

# Define the weighted loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    logits = model(graph, features)
    loss = criterion(logits[train_idx], labels[train_idx])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(graph, features)
        val_loss = criterion(val_logits[val_idx], labels[val_idx])
        val_preds = torch.argmax(val_logits[val_idx], dim=1)
        val_acc = (val_preds == labels[val_idx]).float().mean()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")

# Test the model
model.eval()
with torch.no_grad():
    test_logits = model(graph, features)
    test_preds = torch.argmax(test_logits[test_idx], dim=1)
    print("\nTest Results:")
    print(classification_report(labels[test_idx].numpy(), test_preds.numpy(), target_names=["Non-Fraud", "Fraud"]))