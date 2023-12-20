from abc import ABC, abstractmethod
import numpy as np
from typing import List
from utils import cal_cos_dists_matrix, cal_haversine_dists_matrix
import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch import nn
from SparseConvConfig import SparseConv


class BaseGeoInterpolator(ABC):
    """
    Base class for traditional geospatial interpolation methods like IDW and ADW.
    Initializes with common parameters for neighborhood and distance threshold.
    """

    def __init__(self, n_neighbors=20, cdd=60) -> None:
        """
        :param n_neighbors: Maximum number of neighbors to consider for interpolation.
        :param cdd: Correlation Decay Coefficient,threshold for considering neighbors.
        """
        self.n_neighbors = n_neighbors
        self.cdd = cdd

    def get_interp_ref_points(self, cdist: np.ndarray) -> List[np.ndarray[int]]:
        """
        Selects reference points for interpolation based on cut-off distance and neighbor count.
        :param cdist: Distance matrix between test and training points.
        :return: A list of arrays, each containing indices of the reference points for a test point.
        """
        neighbor_ids = []
        n_neighbors_array = np.sum(cdist < self.cdd, axis=1)

        # Limit the number of neighbors to 'n_neighbors'
        n_neighbors_array[n_neighbors_array > self.n_neighbors] = self.n_neighbors

        # Ensure at least a minimal number of neighbors
        n_neighbors_array[n_neighbors_array < 2] = min(self.n_neighbors // 4, 4)

        # Sort distances and select the top neighbors
        cdist_argsort = np.argsort(cdist, axis=1)
        for i in range(len(cdist_argsort)):
            neighbor_ids.append(cdist_argsort[i][: n_neighbors_array[i]])
        return neighbor_ids

    @abstractmethod
    def get_interp_weights(
        self, train_coords, test_coords, neighbor_ids, cdist
    ) -> List[np.ndarray[float]]:
        """
        compute interpolation weights.
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def interpolate(self, train_coords, test_coords, train_values, cdist) -> np.ndarray:
        """
        Performs the interpolation for test points using training data.
        :param train_coords: Coordinates of training points.
        :param test_coords: Coordinates of test points.
        :param train_values: Observed values at training points.
        :param cdist: Distance matrix between test and training points.
        :return: Interpolated values at test points.
        """
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
    """
    Implements the ADW (Angular Distance Weighted) interpolation method.
    :param m: Decay coefficient controlling weight attenuation with distance.
    """

    def __init__(self, n_neighbors=20, cdd=60, m=4) -> None:
        super().__init__(n_neighbors, cdd)
        self.m = m

    def get_interp_weights(
        self, train_coords, test_coords, neighbor_ids, cdist
    ) -> List[np.ndarray[float]]:
        """
        :return: List of weights corresponding to each reference point set.
        """
        neighbor_weights = []
        for i in range(len(test_coords)):
            # Compute initial weights based on distance
            weights = np.exp(-cdist[i, neighbor_ids[i]] / self.cdd) ** self.m
            weights /= weights.sum()

            # Adjust weights based on spatial distribution
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


class KCN:
    pass


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
