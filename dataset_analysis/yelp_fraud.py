# Analysis of yelp fraud dataset

import dgl
from dgl.data import FraudYelpDataset

# Load the Fraud Yelp dataset
dataset = FraudYelpDataset()

# Access the graph and labels
graph = dataset[0]  # The dataset contains only one graph

# labels = graph.ndata['label']  # Node labels
# features = graph.ndata['feature']  # Node features

# # Print basic information about the dataset
# print("Number of nodes:", graph.num_nodes())
# print("Number of edges:", graph.num_edges())
# print("Feature shape:", features.shape)
# print("Number of classes:", len(set(labels.numpy())))

# # Check a sample feature and label
# print("Sample feature:", features[0])
# print("Sample label:", labels[0])

# Print the available node types and edge types
print("Node types:", graph.ntypes)
print("Edge types:", graph.etypes)

# Fetch and print one review node (since all nodes are of type 'review')
review_node = next(iter(graph.nodes('review')))
print(f"Review Node ID: {review_node}")

# Fetch and print edges for each relation type
for etype in graph.etypes:
    src, dst = graph.edges(etype=etype)
    if len(src) > 0:  # Check if edges exist
        print(f"Edge Type: {etype}")
        print(f"Source Node ID: {src[0]}, Destination Node ID: {dst[0]}")

# Print all node labels (assuming labels are stored in the 'label' attribute)
labels = graph.ndata['label']
print("Node Labels:")
print(labels.numpy())