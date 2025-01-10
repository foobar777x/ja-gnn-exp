import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import FraudYelpDataset
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
dataset = FraudYelpDataset()
graph = dataset[0]

# Extract node features, labels, and train/test masks
features = graph.ndata['feature']
labels = graph.ndata['label']
train_idx = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]

# Apply SMOTE to balance classes in the training set
smote = SMOTE(random_state=42)
X_train, y_train = features[train_idx].numpy(), labels[train_idx].numpy()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Convert back to PyTorch tensors
X_train_resampled = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_resampled = torch.tensor(y_train_resampled, dtype=torch.long)

# Define the Heterogeneous GCN Model
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
        # Return the output for 'review' nodes
        return x['review']

# Initialize the model
in_feats = features.shape[1]
hidden_feats = 16
num_classes = 2  # Binary classification (fraud or non-fraud)
model = HeteroGCN(in_feats, hidden_feats, num_classes, graph.etypes)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    logits = model(graph, features)
    loss = criterion(logits[train_idx], labels[train_idx])  # Use original training graph structure
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(graph, features)
        val_loss = criterion(val_logits[graph.ndata['val_mask']], labels[graph.ndata['val_mask']])
        val_preds = torch.argmax(val_logits[graph.ndata['val_mask']], dim=1)
        val_acc = (val_preds == labels[graph.ndata['val_mask']]).float().mean()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")

# Test the model
model.eval()
with torch.no_grad():
    test_logits = model(graph, features)
    test_preds = torch.argmax(test_logits[graph.ndata['test_mask']], dim=1)
    print("\nTest Results:")
    print(classification_report(labels[graph.ndata['test_mask']].numpy(), test_preds.numpy(), target_names=["Non-Fraud", "Fraud"]))
