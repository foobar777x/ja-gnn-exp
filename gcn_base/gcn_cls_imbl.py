import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import FraudYelpDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
    def __init__(self, in_feats, hidden_feats, num_classes, edge_types):
        super(HeteroGCN, self).__init__()
        self.conv1 = dgl.nn.HeteroGraphConv({
            etype: dgl.nn.GraphConv(in_feats, hidden_feats)
            for etype in edge_types
        })
        self.conv2 = dgl.nn.HeteroGraphConv({
            etype: dgl.nn.GraphConv(hidden_feats, num_classes)
            for etype in edge_types
        })

    def forward(self, g, x):
        # Apply the first layer to each edge type
        x = self.conv1(g, {'review': x})
        x = {k: F.relu(v) for k, v in x.items()}  # Apply activation
        # Apply the second layer
        x = self.conv2(g, x)
        # Only the 'review' node type is used for output
        return x['review']

# Initialize the model
in_feats = features.shape[1]
hidden_feats = 16
num_classes = 2  # Binary classification (fraud or non-fraud)
model = HeteroGCN(in_feats, hidden_feats, num_classes, graph.etypes)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

###############################################################################################
# Dynamic computation of class weights

# Compute class weights (inverse of class frequencies)
num_fraud = (labels == 1).sum().item()
num_non_fraud = (labels == 0).sum().item()
total = num_fraud + num_non_fraud

# Class weights: higher weight for the minority class
class_weights = torch.tensor([total / num_non_fraud, total / num_fraud], dtype=torch.float32)
###############################################################################################

# Define the weighted loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training Loop
num_epochs = 50
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