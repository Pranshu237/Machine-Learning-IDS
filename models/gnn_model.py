import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GCNConv

from config import GNN_EPOCHS, GNN_LR, GNN_NEIGHBOURS, RANDOM_STATE


class GNN(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x)


def build_graph(X, k=GNN_NEIGHBOURS):
    """Connect each flow to the k flows with the most similar features.

    Note: this is a similarity graph, not the network's real structure
    (hosts and the connections between them).
    """
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity")
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    return torch.tensor(X, dtype=torch.float), edge_index


class GNNClassifier:
    """Small wrapper so the GNN can be trained and scored like the other models."""

    def __init__(self, epochs=GNN_EPOCHS, lr=GNN_LR):
        self.epochs = epochs
        self.lr = lr
        self.losses = []

    def fit(self, X, y):
        torch.manual_seed(RANDOM_STATE)
        x, edge_index = build_graph(X)
        self.model = GNN(input_dim=x.shape[1], num_classes=int(np.max(y)) + 1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        target = torch.tensor(np.asarray(y, dtype=int), dtype=torch.long)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = F.cross_entropy(self.model(x, edge_index), target)
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            if (epoch + 1) % 25 == 0 or epoch == 0:
                print(f"GNN epoch {epoch + 1}, loss {loss.item():.4f}")
        return self

    def predict(self, X):
        x, edge_index = build_graph(X)
        self.model.eval()
        with torch.no_grad():
            return self.model(x, edge_index).argmax(dim=1).numpy()
