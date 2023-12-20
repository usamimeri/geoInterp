from typing import List, Tuple
import torch
from torch import nn


class LocalMoranIndex(nn.Module):
    """
    A PyTorch module for computing Local Moran's Index for spatial data.

    This module calculates the Local Moran's Index for each observation in a
    spatial dataset, which measures the degree of spatial autocorrelation.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        X: torch.Tensor,
        neighbor_weights: List[List[float]],
        neighbor_ids: List[List[int]],
    ) -> torch.Tensor:
        """
        Compute the Local Moran's Index for each observation.

        Parameters:
        X (torch.Tensor): A 1D tensor of observations. Shape: (num_nodes,)
        neighbor_weights (List[List[float]]): Weights of the neighboring points
            for each observation. Each sublist corresponds to the weights of
            the neighbors of an observation.
        neighbor_ids (List[List[int]]): Indices of the neighboring points for
            each observation. Each sublist corresponds to the indices of the
            neighbors of an observation.

        Returns:
        torch.Tensor: A 1D tensor of Local Moran's Index values for each observation.
                      Shape: (num_nodes,)

        Example:
        >>> X = torch.tensor([0, 30, 90.0])
        >>> weights = [[0.1, 0.9], [1.0, 0.0], [0.5, 0.5]]
        >>> ids = [[1, 2], [0, 2], [0, 1]]
        >>> lmi = LocalMoranIndex()
        >>> print(lmi(X, weights, ids))
        """
        X_mean = X.mean()
        X_anom = X - X_mean
        X_anom_square = X_anom**2
        moran_indexes = []
        for i in range(len(X)):
            neighbor_id = neighbor_ids[i]
            neighbor_weight = torch.tensor(
                neighbor_weights[i], device=X.device, dtype=X.dtype
            )
            S2_i = torch.sum(neighbor_weight * X_anom_square[neighbor_id]) / (
                len(neighbor_id) - 1
            )
            I_i = (X_anom[i] * torch.sum(neighbor_weight * X_anom[neighbor_id])) / S2_i
            moran_indexes.append(I_i)

        return torch.tensor(moran_indexes, device=X.device)


if __name__ == "__main__":
    X = torch.tensor([0, 30, 90.0]) # 观测值
    weights = [[0.1, 0.9], [0.0, 1.0], [0.5, 0.5]]
    ids = [[1, 2], [0, 2], [0, 1]]
    print(LocalMoranIndex()(X, weights, ids))
