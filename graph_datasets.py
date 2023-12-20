from torch_geometric.data import Dataset, Data
import os
import numpy as np
from utils import read_new_zealand_data, cal_haversine_dists_matrix
from models.interpolators import ADW
from typing import Literal
from tqdm import tqdm
import pandas as pd
import torch


class SparseObsDataset(Dataset):
    def __init__(
        self,
        root,
        processed,
        location: Literal["south", "north", "all"],
        sparsities=[0.1, 0.2, 0.3, 0.4, 0.5],
        n_neighbors=20,
        cdd=60,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.processed = processed
        self.location = location
        self.sparsities = sparsities
        self.n_neighbors = n_neighbors
        self.cdd = cdd
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processed)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return [
            f"{raw_file_name.split('.')[0]}_{self.location}_sparse_{int(sparsity*100)}.pt"
            for raw_file_name in self.raw_file_names
            for sparsity in self.sparsities
        ]

    def process(self):
        for raw_file_path in tqdm(self.raw_paths, desc="Processing"):
            df = read_new_zealand_data(raw_file_path, self.location)
            x, y = self._get_x_y(df, feature_cols=["lat", "lon", "elev"], obs_col="obs")
            for sparsity in self.sparsities:
                train_index, test_index = self._split_indices(
                    len(df), test_size=sparsity
                )
                (
                    edge_index,
                    dists,
                    cdist,
                    train_weights,  # 用于计算莫兰指数的邻居权重
                    train_neighbors,
                ) = self._get_dists_edge_index(
                    x, train_index, test_index, self.n_neighbors, self.cdd
                )
                edge_attr = self._get_edge_attr(x, edge_index, dists)
                data = Data(
                    x=torch.from_numpy(x.astype(np.float32)),
                    y=torch.from_numpy(y.astype(np.float32)),
                    edge_index=torch.from_numpy(edge_index).long(),
                    edge_attr=torch.from_numpy(edge_attr.astype(np.float32)),
                    train_index=train_index,
                    test_index=test_index,
                    dists=torch.from_numpy(dists.astype(np.float32)),
                    cdist=torch.from_numpy(cdist.astype(np.float32)),
                    train_weights=train_weights,
                    train_neighbors=train_neighbors,
                )
                torch.save(
                    data,
                    os.path.join(
                        self.processed_dir,
                        f"{os.path.basename(raw_file_path).split('.')[0]}_{self.location}_sparse_{int(sparsity*100)}.pt",
                    ),
                )

    def _get_x_y(
        self, df: pd.DataFrame, feature_cols=["lat", "lon", "elev"], obs_col="obs"
    ):
        """
        将数据集拆分为特征和标签
        """
        x = df[feature_cols].values
        y = df[obs_col].values
        x = np.concatenate(
            (
                x,
                x[:, 0:2] * np.pi / 180,
                np.cos(x[:, 0:2] * np.pi / 180),
                np.sin(x[:, 0:2] * np.pi / 180),
            ),
            axis=1,
        )

        return x, y

    def _split_indices(self, n_samples, test_size, shuffle=True):
        """
        切分训练集和验证集索引
        """
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        if 0 < test_size < 1:
            # 比例
            threshold = int(n_samples * test_size)
            if threshold <= 0:
                threshold = 1
        elif 1 <= test_size < n_samples:
            # 直接指定测试集数目
            threshold = test_size
        test_index = indices[:threshold]
        train_index = indices[threshold:]
        return train_index, test_index

    def _get_dists_edge_index(self, x, train_index, test_index, n_neighbors, cdd):
        """
        dists:(节点数,节点数) 节点数=训练节点数+测试节点数
        返回和X对应的edge_index，即edge_index的某个索引index和原来的X位置是对应的
        """
        dists = cal_haversine_dists_matrix(x[:, 0:2])  # 全部的矩阵
        cdist = dists[test_index, :][:, train_index]  # 配对矩阵
        adw = ADW(n_neighbors, cdd)
        # 这样会让所有训练节点带一条self_loop
        train_neighbors = adw.get_interp_ref_points(
            dists[train_index, :][:, train_index]
        )

        # 每个测试节点的邻接点 对应原来未经打乱的X的索引 也S就是0代表第一行
        test_neighbors = adw.get_interp_ref_points(cdist)

        # 这里是对应到用于训练的观测值的id 因为我们传入给moran的是data.y[train_index]是经过train_index的变换的，因此我们没必要变回去
        train_moran_neighbors=self.remove_self_neighbor(train_neighbors, list(range(len(train_neighbors))))
        # 用于计算莫兰指数
        train_weights = adw.get_interp_weights(
            x[train_index, 0:2],
            x[train_index, 0:2],
            train_moran_neighbors,
            dists[train_index, :][:, train_index],
        )
        train_neighbors = [train_index[i] for i in train_neighbors]
        test_neighbors = [train_index[i] for i in test_neighbors]
        # 构建源节点目标节点对
        neighbors = [*train_neighbors, *test_neighbors]  # 含每个节点对应邻接节点列表的列表
        indices = np.concatenate((train_index, test_index))  # 全部的索引（可能打乱后）
        # 构建邻接节点连向目标节点的边
        edge_index = np.array(
            [np.concatenate(neighbors), np.repeat(indices, [len(a) for a in neighbors])]
        )
        return (
            edge_index,
            dists,
            cdist,
            train_weights,
            train_moran_neighbors,
        )

    def remove_self_neighbor(self, arr_list, elements_to_remove):
        """
        移除邻居中自己的标号，主要是为了计算莫兰指数时别把自己放进去计算权重
        train_neighbors = [[1, 2, 3], [4, 5]]
        train_index = [1, 5]
        输出: [[2, 3], [4]]
        """
        return [
            [element for element in arr if element != elements_to_remove[i]]
            for i, arr in enumerate(arr_list)
        ]

    def _get_edge_attr(self, X, edge_index, dists):
        """
        用节点的特征差作为边特征
        并加入两个额外特征，分别是两个节点距离和距离的高斯核函数
        """
        num_node_features = X.shape[1]
        source = edge_index[0]
        target = edge_index[1]
        edge_attr = np.zeros((edge_index.shape[1], num_node_features + 2))
        edge_attr[:, :num_node_features] = X[target] - X[source]  # 目标节点和源节点特征差
        edge_attr[:, -2] = dists[source, target]
        edge_attr[:, -1] = np.exp(-edge_attr[:, -2] / 30)  # 距离的高斯核函数
        return edge_attr

    def get(self, idx: int, verbose=False):
        if verbose:
            print("Loading ", self.processed_paths[idx])
        data = torch.load(self.processed_paths[idx])
        return data

    def len(self):
        return len(self.processed_paths)


if __name__ == "__main__":
    dataset = SparseObsDataset("dataset", "sparse_south", "south")
    print(dataset)
