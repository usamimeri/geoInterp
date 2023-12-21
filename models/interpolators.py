from abc import ABC, abstractmethod
import numpy as np
from typing import List
from utils import cal_cos_dists_matrix, cal_haversine_dists_matrix
import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch import nn
from SparseConvConfig import SparseConv
import sklearn
import sklearn.neighbors
import torch_geometric


class BaseGeoInterpolator(ABC):
    def __init__(self, n_neighbors=20, cdd=60) -> None:
        self.n_neighbors = n_neighbors
        self.cdd = cdd

    def get_interp_ref_points(self, cdist: np.ndarray) -> List[np.ndarray[int]]:
        neighbor_ids = []
        n_neighbors_array = np.sum(cdist < self.cdd, axis=1)
        n_neighbors_array[n_neighbors_array > self.n_neighbors] = self.n_neighbors
        n_neighbors_array[n_neighbors_array <= 2] = max(self.n_neighbors // 4, 4)
        cdist_argsort = np.argsort(cdist, axis=1)
        for i in range(len(cdist_argsort)):
            neighbor_ids.append(cdist_argsort[i][: n_neighbors_array[i]])
        return neighbor_ids

    @abstractmethod
    def get_interp_weights(
        self, train_coords, test_coords, neighbor_ids, cdist
    ) -> List[np.ndarray[float]]:
        raise NotImplementedError

    def interpolate(self, train_coords, test_coords, train_values, cdist) -> np.ndarray:
        neighbor_ids = self.get_interp_ref_points(cdist)
        neighbor_weights = self.get_interp_weights(
            train_coords, test_coords, neighbor_ids, cdist
        )
        test_values = [
            np.dot(train_values[neighbor_ids[i]], neighbor_weights[i])
            for i in range(len(test_coords))
        ]
        return np.array(test_values)


class ADW(BaseGeoInterpolator):
    def __init__(self, n_neighbors=20, cdd=60, m=4) -> None:
        super().__init__(n_neighbors, cdd)
        self.m = m

    def get_interp_weights(
        self, train_coords, test_coords, neighbor_ids, cdist
    ) -> List[np.ndarray[float]]:
        neighbor_weights = []
        for i in range(len(test_coords)):
            weights = np.exp(-cdist[i, neighbor_ids[i]] / self.cdd) ** self.m
            weights /= weights.sum()

            cosine_matrix = cal_cos_dists_matrix(
                train_coords[neighbor_ids[i]], origin=test_coords[i]
            )
            a_i = np.dot(cosine_matrix, weights.reshape(-1, 1)).flatten()
            w_sum = np.full_like(weights, 1.0) - weights
            w_sum[w_sum <= 0.0] = 1e-4
            weights = weights * (1 + a_i / w_sum)
            weights /= weights.sum()
            neighbor_weights.append(weights)
        return neighbor_weights


class GATModel(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(num_node_features, 48)
        self.conv2 = GATConv(48, 60)
        self.conv3 = GATConv(60, 1)

    def forward(self, x, edge_index, edge_attr, *args):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index)
        return x.flatten()


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 48)
        self.conv2 = GCNConv(48, 60)
        self.conv3 = GCNConv(60, 1)

    def forward(self, x, edge_index, *args):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x.flatten()


class AGAIN(nn.Module):
    def __init__(
        self,
        in_dim,
        h1_dim,
        h2_dim=25,
        threshold=0.03,
        scale=30,
        edge_dim=None,
        dropout=0.3,
        num_heads=8,
    ):
        super(AGAIN, self).__init__()
        self.individual1 = nn.Linear(in_dim, h1_dim)
        self.individual2 = nn.Linear(h1_dim, 1)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.bn3 = nn.BatchNorm1d(in_dim)
        self.bn4 = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.num_heads = num_heads
        if scale:
            scale = 1 / torch.log(torch.Tensor([scale]))
        else:
            scale = torch.Tensor([1])

        self.sparseConv1 = SparseConv(
            in_channels=in_dim,
            out_channels=h1_dim,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            threshold=threshold,
            idx=0,
            scale=scale,
        )

        self.sparseConv2 = SparseConv(
            in_channels=in_dim,
            out_channels=h2_dim,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            threshold=threshold,
            idx=1,
            scale=scale,
        )

        self.fc1 = nn.Linear(h1_dim * num_heads, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)
        self.fc3 = nn.Linear(h2_dim * num_heads, in_dim)
        self.fc4 = nn.Linear(in_dim, in_dim)
        self.fc5 = nn.Linear(in_dim, 1)
        self.elu = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, x, edge_index, edge_attr, sparse_train=True):
        self.sparseConv1.sparse_train = sparse_train
        self.sparseConv2.sparse_train = sparse_train
        x_src = self.sparseConv1(
            x=x, edge_index=edge_index, edge_attr=edge_attr
        )  # h1_dim*num_heads
        x_src = self.elu(self.fc1(x_src))  # in_dim
        x_src += x  # in_dim
        x_src = self.bn1(x_src)  # in_dim

        x_src = self.elu(self.fc2(x_src))  # in_dim
        x_src += x  # in_dim
        x_src = self.bn2(x_src)  # in_dim

        x_src = self.sparseConv2(
            x=x_src, edge_index=edge_index, edge_attr=edge_attr
        )  # h2_dim*num_heads
        x_src = self.elu(self.fc3(x_src))  # in_dim
        x_src += x  # in_dim
        x_src = self.bn3(x_src)  # in_dim

        x_src = self.elu(self.fc4(x_src))  # in_dim
        x_src += x  # in_dim
        x_src = self.bn4(x_src)  # in_dim
        y = self.fc5(x_src).flatten()  # 1
        return y


class KCN(torch.nn.Module):
    def __init__(self, n_neighbors, device, data) -> None:
        super().__init__()

        self.all_coords = data.x[:, 0:2]
        self.all_features = data.x[:, 2:]
        self.y = data.y.reshape(-1, 1)
        self.device = device
        # set neighbor relationships within the training set
        self.n_neighbors = n_neighbors
        self.knn = sklearn.neighbors.NearestNeighbors(n_neighbors=self.n_neighbors).fit(
            self.all_coords
        )
        distances, self.train_neighbors = self.knn.kneighbors(
            None, return_distance=True
        )
        self.length_scale = np.median(distances.flatten())

        with torch.no_grad():
            self.graph_inputs = []
            for i in range(self.all_coords.shape[0]):
                # 为每一个点构造子图
                att_graph = self.form_input_graph(
                    self.all_coords[i],
                    self.all_features[i],
                    self.train_neighbors[i],
                )
                self.graph_inputs.append(att_graph)

        input_dim = self.all_features.shape[1] + 2
        output_dim = self.y.shape[1]

        self.gnn = GNN(input_dim).to(self.device)

        # the last linear layer
        self.linear = torch.nn.Linear(60, output_dim, bias=False)
        self.last_activation = torch.nn.ReLU()

        self.collate_fn = torch_geometric.loader.dataloader.Collater(None, None)

    def forward(self, indices):
        batch_inputs = []
        for i in indices:
            # 读取预先计算好的子图
            batch_inputs.append(self.graph_inputs[i])

        batch_inputs = self.collate_fn(batch_inputs)
        batch_inputs = batch_inputs.to(self.device)
        # run gnn on the graph input
        output = self.gnn(
            batch_inputs.x, batch_inputs.edge_index, batch_inputs.edge_attr
        )
        # take representations only corresponding to center nodes
        output = torch.reshape(output, [-1, (self.n_neighbors + 1), output.shape[1]])
        center_output = output[:, 0]
        pred = self.last_activation(self.linear(center_output))

        return pred

    def form_input_graph(self, coord, feature, neighbors):
        output_dim = self.y.shape[1]

        y = torch.concat([torch.zeros([1, output_dim]), self.y[neighbors]], axis=0)

        indicator = torch.zeros([neighbors.shape[0] + 1])
        indicator[0] = 1.0

        features = torch.concat(
            [feature[None, :], self.all_features[neighbors]], axis=0
        )

        graph_features = torch.concat([features, y, indicator[:, None]], axis=1)

        all_coords = torch.concat([coord[None, :], self.all_coords[neighbors]], axis=0)

        kernel = sklearn.metrics.pairwise.rbf_kernel(
            all_coords.numpy(), gamma=1 / (2 * self.length_scale**2)
        )

        adj = torch.from_numpy(kernel)
        nz = adj.nonzero(as_tuple=True)
        edges = torch.stack(nz, dim=0)
        edge_weights = adj[nz]

        attributed_graph = torch_geometric.data.Data(
            x=graph_features, edge_index=edges, edge_attr=edge_weights, y=None
        )

        return attributed_graph

    def _normalize_adj(self, adj):
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        adj_normalized = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]

        return adj_normalized


class GNN(torch.nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()

        self.hidden_sizes = [48, 60]
        self.dropout = 0.1

        conv_layer = torch_geometric.nn.GCNConv(
            input_dim, self.hidden_sizes[0], bias=False, add_self_loops=True
        )

        self.add_module("layer0", conv_layer)

        for ilayer in range(1, len(self.hidden_sizes)):
            conv_layer = torch_geometric.nn.GCNConv(
                self.hidden_sizes[ilayer - 1],
                self.hidden_sizes[ilayer],
                bias=False,
                add_self_loops=True,
            )

            self.add_module("layer" + str(ilayer), conv_layer)

    def forward(self, x, edge_index, edge_weight):
        for conv_layer in self.children():
            x = conv_layer(x, edge_index, edge_weight=edge_weight)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        return x


if __name__ == "__main__":
    train_coords = np.array([[10, 10], [20, 20]])
    test_coords = np.array([[10, 10]])
    cdist = cal_haversine_dists_matrix(test_coords, train_coords)
    train_values = np.array([20, 100])
    adw = ADW(cdd=1000)
    neighbor_ids = adw.get_interp_ref_points(cdist)
    print(cdist[0, neighbor_ids[0]])
    print(train_coords[neighbor_ids[0]])
    print(adw.get_interp_weights(train_coords, test_coords, neighbor_ids, cdist))
    print(adw.interpolate(train_coords, test_coords, train_values, cdist))
