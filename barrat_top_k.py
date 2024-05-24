import networkx as nx
import itertools

def barrats_coefficient(G, node, neighbors):
    """
    Calculate Barrat's weighted clustering coefficient for a given node and its subset of neighbors.
    """
    s_i = sum(G[node][nbr]['weight'] for nbr in G.neighbors(node))
    k_i = len(neighbors)
    if k_i < 2:
        return 0  # No triangles possible if less than 2 neighbors
    
    triangles_sum = 0
    for j, k in itertools.combinations(neighbors, 2):
        if G.has_edge(j, k):
            w_ij = G[node][j]['weight']
            w_ik = G[node][k]['weight']
            w_jk = G[j][k]['weight']
            triangles_sum += ((w_ij + w_ik) / 2) * w_jk
    
    return triangles_sum / (s_i * k_i * (k_i - 1))

def find_optimal_k(G, target_node, max_k):
    """
    Determine the optimal number of neighbors k by maximizing Barrat's weighted clustering coefficient.
    """
    neighbors = list(G.neighbors(target_node))
    max_coefficient = 0
    optimal_k = 0
    optimal_neighbors = []

    # Iterate over all combinations of neighbor subsets up to size max_k
    for k in range(1, min(max_k, len(neighbors)) + 1):
        for combination in itertools.combinations(neighbors, k):
            coefficient = barrats_coefficient(G, target_node, combination)
            if coefficient > max_coefficient:
                max_coefficient = coefficient
                optimal_k = k
                optimal_neighbors = combination

    return optimal_k, max_coefficient, optimal_neighbors

# Transaction Graph
G = nx.Graph()
# Add nodes, edges, and weights
edges = global.edges
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# Find the optimal k for node 1
target_node = 1
max_k = 3  # Maximum number of neighbors to consider

optimal_k, max_coefficient, optimal_neighbors = find_optimal_k(G, target_node, max_k)
print(f"Optimal k: {optimal_k}")
print(f"Maximum Clustering Coefficient: {max_coefficient}")
print(f"Top k neighbors: {optimal_neighbors}")
