from abc import ABC, abstractmethod
import numpy as np
from typing import List
from utils import cal_cos_dists_matrix, cal_haversine_dists_matrix


class BaseGeoInterpolator(ABC):
    @abstractmethod
    def get_interp_ref_points(self):
        pass

    @abstractmethod
    def get_interp_weights(self):
        pass

    @abstractmethod
    def interpolate(self, train_coords, test_coords, train_values, cdist):
        assert len(train_coords) == len(train_values), "训练集节点数和观测值数需要一致"
        assert cdist.shape[0] == len(test_coords)
        assert cdist.shape[1] == len(train_coords)


class ADW(BaseGeoInterpolator):
    """
    计算ADW插值法
    train_coords:训练集点 (N,2)
    test_coords:测试集点(M,2)
    train_values:训练集观测值(N,)
    cdist:测试集训练集距离矩阵(N,M)

    """

    def __init__(self, n_neighbors=20, cdd=60, m=4) -> None:
        self.n_neighbors = n_neighbors
        self.cdd = cdd
        self.m = m  # 衰减系数 越大随距离衰减越快

    def get_interp_ref_points(self, cdist) -> List[np.ndarray[int]]:
        """
        根据cdd筛选和邻居数，获取用于插值测试点的参考点集
        """
        neighbor_ids = []
        n_neighbors_array = np.sum(cdist < self.cdd, axis=1)  # 每个测试点对应的参考点个数
        n_neighbors_array[
            n_neighbors_array > self.n_neighbors
        ] = self.n_neighbors  # 如果很多小于cdd，控制数目在预设的邻居数下

        n_neighbors_array[n_neighbors_array < 2] = min(
            self.n_neighbors // 4, 2
        )  # 如果出现了空参考点,则取一定的邻居
        cdist_argsort = np.argsort(cdist, axis=1)  # 按行升序排列索引
        for i in range(len(cdist_argsort)):
            # 对排序好的索引，选择对应的参考点数加入索引集
            neighbor_ids.append(cdist_argsort[i][: n_neighbors_array[i]])
        return neighbor_ids

    def get_interp_weights(
        self, train_coords, test_coords, neighbor_ids, cdist
    ) -> List[np.ndarray[float]]:
        """
        获取对应的参考点权重
        返回和neighbor_ids中的每个参考点列表一一对应的权重
        比如[2,0,1]对应训练点中的行索引
        则权重[0.3,0.4,0.3]代表 2 对应权重0.3 0对应权重0.4 1对应权重0.3
        """
        neighbor_weights = []
        for i in range(len(test_coords)):
            weights = np.exp(-cdist[i, neighbor_ids[i]] / self.cdd) ** self.m
            weights /= weights.sum()
            cosine_matrix = cal_cos_dists_matrix(
                train_coords[neighbor_ids[i]], origin=test_coords[i]
            )
            a_i = np.dot(cosine_matrix, weights.reshape(-1, 1)).flatten()  # 计算余弦距离和权重内积
            # a_i 每个元素对应该索引对应计算式的分子\sum w_k(1-cos(θ))部分
            w_sum = (
                np.full_like(weights, weights.sum()) - weights
            )  # 计算分母，因为k不等于i所以得去减去本身
            weights = weights * (1 + a_i / w_sum)
            weights /= weights.sum()
            neighbor_weights.append(weights)
        return neighbor_weights

    def interpolate(self, train_coords, test_coords, train_values, cdist):
        neighbor_ids = self.get_interp_ref_points(cdist)
        neighbor_weights = self.get_interp_weights(
            train_coords, test_coords, neighbor_ids, cdist
        )
        test_values = []
        for i in range(len(test_coords)):
            interp_value = np.dot(train_values[neighbor_ids[i]], neighbor_weights[i])
            test_values.append(interp_value)

        return np.array(test_values)


if __name__ == "__main__":
    train_coords = np.array([[0, 10], [89, 170]])
    test_coords = np.array([[10, 10]])
    cdist = cal_haversine_dists_matrix(test_coords, train_coords)
    train_values = np.array([20, 100])
    adw = ADW(cdd=1000)
    neighbor_ids = adw.get_interp_ref_points(cdist)
    print(cdist[0, neighbor_ids[0]])
    print(train_coords[neighbor_ids[0]])
    print(adw.get_interp_weights(train_coords, test_coords, neighbor_ids, cdist))
    print(adw.interpolate(train_coords, test_coords, train_values, cdist))
