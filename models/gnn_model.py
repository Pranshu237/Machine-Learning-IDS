import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.neighbors import kneighbors_graph
import numpy as np

class GNN(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.fc(x)

def build_graph(X):
    A = kneighbors_graph(X, n_neighbors=5, mode='connectivity')
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    x = torch.tensor(X, dtype=torch.float)
    return x, edge_index