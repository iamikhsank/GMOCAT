import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class GraphAggregator(nn.Module):
    """ 
    Implements Graph Relation Aggregator using GAT (Graph Attention Network). 
    Aggregates information from Questions and Concepts. 
    """ 
    def __init__(self, n_nodes, dim, device): 
        super(GraphAggregator, self).__init__() 
        if not HAS_PYG: raise ImportError("PyG required") 
        self.device = device
         
        self.node_emb = nn.Embedding(n_nodes, dim) 
        self.gat1 = GATConv(dim, dim, heads=2, dropout=0.2) 
        self.gat2 = GATConv(dim * 2, dim, heads=1, concat=False, dropout=0.2) 
         
    def forward(self, edge_index): 
        x = self.node_emb.weight 
        x = F.elu(self.gat1(x, edge_index)) 
        x = F.elu(self.gat2(x, edge_index)) 
        return x # [n_nodes, dim] 

def build_graph(q_matrix, device): 
    """Constructs bipartite graph (Questions <-> Concepts)""" 
    n_q, n_c = q_matrix.shape 
    edges = [] 
    rows, cols = q_matrix.nonzero(as_tuple=True) 
    for q_idx, c_idx in zip(rows, cols): 
        c_node = n_q + c_idx.item() # Offset concept IDs 
        q_node = q_idx.item() 
        edges.append([q_node, c_node]) 
        edges.append([c_node, q_node]) # Undirected 
     
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device) 
    return edge_index, n_q + n_c 
