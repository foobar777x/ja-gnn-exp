import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dgl.data import FraudYelpDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create directory for saving edge weights and subgraphs
EDGE_WEIGHTS_DIR = "edge_weights"
os.makedirs(EDGE_WEIGHTS_DIR, exist_ok=True)


# Utility Functions
def normalize_edge_weights(weights):
    """Normalize edge weights to range [0, 1]."""
    min_val, max_val = weights.min(), weights.max()
    if max_val > min_val:
        return (weights - min_val) / (max_val - min_val)
    else:
        return torch.zeros_like(weights)


def load_edge_weights(filename):
    """Load edge weights from a file."""
    filepath = os.path.join(EDGE_WEIGHTS_DIR, filename)
    if os.path.exists(filepath):
        return torch.load(filepath)
    return None


def save_subgraph(subgraph, filename):
    """Save subgraph to a file."""
    filepath = os.path.join(EDGE_WEIGHTS_DIR, filename)
    torch.save(subgraph, filepath)
    print(f"Subgraph saved to {filepath}")


def create_subgraph(graph, edge_type, subgraph_name):
    """Create a subgraph for a given edge type (e.g., R-U-R)."""
    filename = f"{subgraph_name}_subgraph_weights.pt"
    subgraph_filepath = os.path.join(EDGE_WEIGHTS_DIR, filename)
    
    # Skip creation if subgraph file exists
    if os.path.exists(subgraph_filepath):
        print(f"Subgraph file {subgraph_filepath} already exists. Skipping subgraph creation.")
        return

    edge_weights = load_edge_weights(f"{subgraph_name}_weights.pt")
    if edge_weights is None:
        raise ValueError(f"{subgraph_name} edge weights not calculated. Run assign_{subgraph_name}_weights first.")

    src, dst = graph.edges(etype=edge_type)
    new_src, new_dst, new_weights = [], [], []

    for i in tqdm(range(len(src)), desc=f"Constructing {subgraph_name} subgraph"):
        for u in graph.predecessors(src[i], etype=edge_type):
            for r2 in graph.successors(u, etype=edge_type):
                if r2 != src[i]:  # Avoid self-loops unless required
                    new_src.append(src[i].item())
                    new_dst.append(r2.item())
                    edge_idx = graph.edge_ids(u, r2, etype=edge_type)
                    weight = edge_weights[i] + edge_weights[edge_idx]
                    new_weights.append(weight)

    # Normalize weights
    new_weights = torch.tensor(new_weights)
    new_weights = normalize_edge_weights(new_weights)

    # Create new DGLGraph
    subgraph = dgl.graph((torch.tensor(new_src), torch.tensor(new_dst)))
    subgraph.edata['weight'] = new_weights

    # Save the subgraph
    save_subgraph(subgraph, filename)


def assign_edge_weights(graph, edge_type, filename):
    """Assign and save edge weights for a specific edge type."""
    edge_weights = load_edge_weights(filename)
    if edge_weights is not None:
        graph.edges[edge_type].data['weight'] = edge_weights
        return

    features = graph.ndata['feature']
    src, dst = graph.edges(etype=edge_type)
    edge_weights = torch.zeros(graph.num_edges(edge_type))

    for i in tqdm(range(len(src)), desc=f"Calculating {edge_type} edge weights"):
        src_feat = features[src[i]]
        dst_feat = features[dst[i]]
        cos_sim = F.cosine_similarity(src_feat.unsqueeze(0), dst_feat.unsqueeze(0)).item()
        edge_weights[i] = (cos_sim + 1) / 2

    edge_weights = normalize_edge_weights(edge_weights)
    graph.edges[edge_type].data['weight'] = edge_weights
    torch.save(edge_weights, os.path.join(EDGE_WEIGHTS_DIR, filename))


# JAGNN Layer
class JAGNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.1, attn_drop=0.1):
        super(JAGNNLayer, self).__init__()
        self.out_feats = out_feats  # Save for later use
        self.gat = nn.ModuleDict({
            etype: dgl.nn.GATConv(
                in_feats, out_feats // num_heads, num_heads,
                feat_drop=feat_drop, attn_drop=attn_drop
            )
            for etype in ['net_rur', 'net_rsr', 'net_rtr']
        })
        self.top5_linear = nn.Linear(out_feats, 8)  # Reduce top-5 embedding to size 8
        self.bottom2_linear = nn.Linear(out_feats, 8)  # Reduce bottom-2 embedding to size 8
        self.proj = nn.Linear(out_feats + 8 + 8, out_feats)  # Project concatenated embedding back to out_feats

    def forward(self, g, h, edge_weights, subgraph_names):
        gat_outs = []
        zero_vector = torch.zeros(self.out_feats, device=h.device)  # Use the saved out_feats

        for etype in g.etypes:
            # Generate embeddings using GAT
            subgraph_etype = dgl.edge_type_subgraph(g, [etype])
            edge_weight = subgraph_etype.edges[etype].data['weight']
            gat_out = self.gat[etype](subgraph_etype, h, edge_weight=edge_weight)

            # Load the subgraph for the current edge type
            subgraph_name = subgraph_names.get(etype)
            if not subgraph_name:
                raise ValueError(f"Subgraph name not found for edge type: {etype}")
            subgraph_file = os.path.join(EDGE_WEIGHTS_DIR, f"{subgraph_name}_subgraph_weights.pt")

            if os.path.exists(subgraph_file):
                subgraph = torch.load(subgraph_file)
            else:
                print(f"Subgraph file for {etype} not found: {subgraph_file}. Using zero vector for all nodes.")
                subgraph = None  # No subgraph available

            # Compute enhanced embeddings using the subgraph
            enhanced_embeddings = []
            for node in range(g.num_nodes()):
                if subgraph and node in subgraph.nodes():  # Check if the node is in the subgraph
                    neighbors = subgraph.successors(node)
                    if len(neighbors) > 0:
                        neighbor_weights = subgraph.edata['weight'][subgraph.edge_ids(node, neighbors)]
                        neighbor_embeddings = gat_out[neighbors]

                        # Sort neighbors by edge weight
                        sorted_indices = torch.argsort(neighbor_weights, descending=True)
                        top_5 = neighbor_embeddings[sorted_indices[:5]].mean(dim=0).squeeze(dim=0) if len(sorted_indices) >= 1 else zero_vector
                        bottom_2 = neighbor_embeddings[sorted_indices[-2:]].mean(dim=0).squeeze(dim=0) if len(sorted_indices) >= 2 else zero_vector
                    else:
                        top_5, bottom_2 = zero_vector, zero_vector
                else:
                    top_5, bottom_2 = zero_vector, zero_vector  # Use zero vector if no subgraph or no neighbors

                # Reduce dimensions of top-5 and bottom-2 embeddings
                top_5 = self.top5_linear(top_5)
                bottom_2 = self.bottom2_linear(bottom_2)

                # Concatenate and project
                concat_embedding = torch.cat([gat_out[node].view(-1), top_5, bottom_2], dim=0)
                enhanced_embedding = self.proj(concat_embedding)  # Project back to out_feats
                enhanced_embeddings.append(enhanced_embedding)
            
            enhanced_embeddings = torch.stack(enhanced_embeddings)
            gat_outs.append(enhanced_embeddings)

        return torch.cat(gat_outs, dim=1)


# JAGNN Model
class JAGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, num_heads, num_layers):
        super(JAGNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_feats if i == 0 else hidden_feats * len(['net_rur', 'net_rsr', 'net_rtr'])
            self.layers.append(JAGNNLayer(in_dim, hidden_feats, num_heads))
        self.fc = nn.Linear(hidden_feats * len(['net_rur', 'net_rsr', 'net_rtr']), num_classes)

    def forward(self, g, h, subgraph_names):
        edge_weights = {etype: g.edges[etype].data['weight'] for etype in g.etypes}
        for layer in self.layers:
            h = layer(g, h, edge_weights, subgraph_names)  # Pass subgraph_names to each layer
        return self.fc(h)


# Main Script
if __name__ == "__main__":
    dataset = FraudYelpDataset()
    graph = dataset[0]

    # Subgraph names mapping
    subgraph_names = {'net_rur': 'rur', 'net_rsr': 'rsr', 'net_rtr': 'rtr'}

    # Assign edge weights
    assign_edge_weights(graph, 'net_rur', "rur_weights.pt")
    assign_edge_weights(graph, 'net_rsr', "rsr_weights.pt")
    assign_edge_weights(graph, 'net_rtr', "rtr_weights.pt")

    # Create subgraphs
    create_subgraph(graph, 'net_rur', 'rur')
    # create_subgraph(graph, 'net_rsr', 'rsr')
    # create_subgraph(graph, 'net_rtr', 'rtr')

    # Add self-loops
    for etype in graph.etypes:
        graph = dgl.add_self_loop(graph, etype=etype)

    # Extract features and labels
    features = graph.ndata['feature']
    labels = graph.ndata['label']

    # Split data
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

    # Initialize and train model
    in_feats = features.shape[1]
    num_heads = 1
    hidden_feats = 16
    num_classes = 2
    num_layers = 2

    model = JAGNN(in_feats, hidden_feats, num_classes, num_heads, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    class_weights = torch.tensor([1.0, 10.0], dtype=torch.float32)  # Example weights, adjust as needed
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    for epoch in range(100):
        model.train()
        logits = model(graph, features, subgraph_names)
        loss = criterion(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(graph, features, subgraph_names)
            val_loss = criterion(val_logits[val_idx], labels[val_idx])
            val_preds = torch.argmax(val_logits[val_idx], dim=1)
            val_acc = (val_preds == labels[val_idx]).float().mean()

        print(f"Epoch {epoch + 1}: Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(graph, features, subgraph_names)
        test_preds = torch.argmax(test_logits[test_idx], dim=1)
        print("\nTest Results:")
        print(classification_report(labels[test_idx].cpu().numpy(), test_preds.cpu().numpy(), target_names=["Non-Fraud", "Fraud"]))