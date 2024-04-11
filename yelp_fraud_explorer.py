
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.data

dataset = dgl.data.FraudYelpDataset()
print('Number of categories:', dataset.num_classes)

graph = dataset[0]

print("Nodes: ", graph.ndata['feature'])
print("Labels: ", graph.ndata['label'])

# print("Edges: ", graph.edata)   # No edge features in this graph

print("Length of input features: ", graph.ndata['feature'].shape[1])

print("Node types: ", graph.ntypes)
print("Edge types: ", graph.etypes)

# graph = dgl.to_homogeneous(graph)  # Making the graph homogeneous

# print("Homogeneous Edge types: ", graph.etypes)

# print("Features: ", graph.ndata)


from dgl.nn import HeteroGraphConv, GraphConv

class HeteroGCN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, num_classes):

        super(HeteroGCN, self).__init__()  # Crucial to call the constructor of the parent class (nn.Module does a lot of preprocessing in its constructor)
        
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, h1_feats) for rel in graph.etypes
        }, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(h1_feats, h2_feats) for rel in graph.etypes
        }, aggregate='sum')
        self.conv3 = HeteroGraphConv({
            rel: GraphConv(h2_feats, num_classes) for rel in graph.etypes
        }, aggregate='sum')

        # self.conv1 = GraphConv(in_feats, h1_feats)
        # self.conv2 = GraphConv(h1_feats, h2_feats)
        # self.conv3 = GraphConv(h2_feats, num_classes)

    def forward(self, graph, in_feat):
        h = self.conv1(graph, in_feat)
        # h = F.relu(h)
        h = {ntype: F.relu(h_feat) for ntype, h_feat in h.items()}
        h = self.conv2(graph, h)
        # h = F.relu(h)
        h = {ntype: F.relu(h_feat) for ntype, h_feat in h.items()}
        h = self.conv3(graph, h)
        return h

model = HeteroGCN(graph.ndata['feature'].shape[1], 16, 8, dataset.num_classes)

def train(graph, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = graph.ndata['feature']
    labels =  graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    features_by_type = {ntype: graph.nodes[ntype].data['feature'] for ntype in graph.ntypes}

    for e in range(500):

        logits = model(graph, features_by_type)   # __call__ underneath the hood for nn.Module calls forward method

        logits = logits['review']

        pred = logits.argmax(1)

        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Accuracy computation
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('We are now running epoch {}, currently loss: {:.3f} val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format( 
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

model = HeteroGCN(graph.ndata['feature'].shape[1], 16, 8, dataset.num_classes)
train(graph, model)
