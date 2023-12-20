from typing import List, Tuple
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class LocalMoranIndex(nn.Module):
    """
    A PyTorch module for computing Local Moran's Index for spatial data.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        y: torch.Tensor,
        neighbor_weights: List[List[float]],
        neighbor_ids: List[List[int]],
    ) -> torch.Tensor:
        """
        Compute the Local Moran's Indey for each observation.
        Returns:
        torch.Tensor: A 1D tensor of Local Moran's Index values for each observation.
                      Shape: (num_nodes,)
        """

        valid_len = torch.tensor([len(i) for i in neighbor_weights], device=y.device)
        # [[1.0],[0.3,0.4,0.3]]->[[1.0,0,0,0.0],[0.3,0.4,0.3]]
        weights = pad_sequence([torch.tensor(i) for i in neighbor_weights]).T.to(
            y.device
        )  # w_{ij}
        ids = pad_sequence([torch.tensor(i).long() for i in neighbor_ids]).T.to(
            y.device
        )
        y_mean = y.mean()
        Z = y[ids] - y_mean  # y_j-\bar y
        S2 = torch.sum(weights * (Z**2), dim=1) / (valid_len - 1)
        I = ((y - y_mean) * torch.sum(weights * Z, dim=1)) / S2
        return torch.mean(torch.abs(I))


if __name__ == "__main__":
    y = torch.tensor([0, 30, 90.0], requires_grad=True)  # 观测
    weights = [[0.1, 0.9], [0, 0.8, 0.2], [0.5, 0.5]]
    ids = [[1, 2], [0, 2, 1], [0, 1]]
    moran = LocalMoranIndex()
    output = moran(y, weights, ids)
    output.backward()
    print(y.grad)
