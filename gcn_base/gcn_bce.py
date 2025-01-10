import dgl
from dgl.data import FraudYelpDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report

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

# Define a Heterogeneous GCN Model
class HeteroGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, edge_types):
        super(HeteroGCN, self).__init__()
        self.conv1 = dgl.nn.HeteroGraphConv({
            etype: dgl.nn.GraphConv(in_feats, hidden_feats)
            for etype in edge_types
        })
        self.conv2 = dgl.nn.HeteroGraphConv({
            etype: dgl.nn.GraphConv(hidden_feats, 1)  # Single output for binary classification
            for etype in edge_types
        })

    def forward(self, g, x):
        # Apply the first layer to each edge type
        x = self.conv1(g, {'review': x})
        x = {k: F.relu(v) for k, v in x.items()}  # Apply activation
        # Apply the second layer
        x = self.conv2(g, x)
        # Only the 'review' node type is used for output
        return x['review'].squeeze(1)  # Squeeze for single logit output per node

# Initialize the model
in_feats = features.shape[1]
hidden_feats = 16
model = HeteroGCN(in_feats, hidden_feats, graph.etypes)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Dynamic computation of class weights
num_fraud = (labels == 1).sum().item()
num_non_fraud = (labels == 0).sum().item()
total = num_fraud + num_non_fraud
pos_weight = torch.tensor([num_non_fraud / num_fraud], dtype=torch.float32)

# Define the binary cross-entropy loss with weights
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    logits = model(graph, features)
    loss = criterion(logits[train_idx], labels[train_idx].float())  # BCE requires float labels
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(graph, features)
        val_loss = criterion(val_logits[val_idx], labels[val_idx].float())
        val_preds = torch.sigmoid(val_logits[val_idx]) > 0.5  # Default threshold
        val_acc = (val_preds == labels[val_idx]).float().mean()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")

# Find the optimal threshold on the validation set
model.eval()
with torch.no_grad():
    val_logits = model(graph, features)
    val_probs = torch.sigmoid(val_logits[val_idx]).detach().numpy()  # Convert logits to probabilities
    val_labels = labels[val_idx].numpy()

thresholds = np.arange(0.1, 1.0, 0.05)
best_threshold = 0.5
best_f1 = 0.0

print("\nThreshold Tuning on Validation Set:")
for threshold in thresholds:
    val_preds = (val_probs > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
    print(f"Threshold: {threshold:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nOptimal Threshold: {best_threshold:.2f} with F1-score: {best_f1:.4f}")

# Test the model using the optimal threshold
with torch.no_grad():
    test_logits = model(graph, features)
    test_probs = torch.sigmoid(test_logits[test_idx]).detach().numpy()
    test_preds = (test_probs > best_threshold).astype(int)
    print("\nTest Results:")
    print(classification_report(labels[test_idx].numpy(), test_preds, target_names=["Non-Fraud", "Fraud"]))