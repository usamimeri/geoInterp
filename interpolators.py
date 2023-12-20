from abc import ABC, abstractmethod
import numpy as np
from typing import List
from utils import cal_cos_dists_matrix, cal_haversine_dists_matrix


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
        n_neighbors_array = np.sum(
            cdist < self.cdd, axis=1
        )  

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


class KCN:
    pass


class GAT:
    pass


class AGAIN:
    pass


class GCN:
    pass


if __name__ == "__main__":
    train_coords = np.array([[10,10], [20, 20]])
    test_coords = np.array([[10, 10]])
    cdist = cal_haversine_dists_matrix(test_coords, train_coords)
    train_values = np.array([20, 100])
    adw = ADW(cdd=1000)
    neighbor_ids = adw.get_interp_ref_points(cdist)
    print(cdist[0, neighbor_ids[0]])
    print(train_coords[neighbor_ids[0]])
    print(adw.get_interp_weights(train_coords, test_coords, neighbor_ids, cdist))
    print(adw.interpolate(train_coords, test_coords, train_values, cdist))
