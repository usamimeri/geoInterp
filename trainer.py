from torch import nn
import torch
import torch_geometric
from models.interpolators import ADW
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss(reduction="mean").to(device)


class StatisticInterpTrainer:
    """
    传统统计方法类的推理，由于是统计方法没有学习率之类训练参数

    - 节点特征x：`(节点数,特征数)`（没有打乱）
        - 节点标签y：`(节点数,)`（没有打乱）
        - 训练集索引：train_index`(训练节点数,)`
        - 测试集索引：test_index`(测试节点数,)`
        - 配对距离矩阵cdists：`（测试节点数，训练节点数）`

    输出：
    1. 测试集上的验证集损失RMSE
    2. dataframe，字段为：测试集索引，纬度、经度、真实值、模型预测值
    """

    def __init__(
        self,
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        train_index: torch.Tensor,
        test_index: torch.Tensor,
        cdist: torch.Tensor,
    ) -> None:
        self.train_coords = x[train_index, 0:2].numpy()
        self.test_coords = x[test_index, 0:2].numpy()
        self.cdist = cdist.numpy()
        self.train_index = train_index.numpy()
        self.test_index = test_index.numpy()
        self.model = model
        self.y = y.numpy()

    def calculate(self):
        y_true = self.y[self.test_index]
        y_pred = self.model.interpolate(
            self.train_coords, self.test_coords, self.y[self.train_index], self.cdist
        )
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        df = pd.DataFrame(columns=["test_index", "lat", "lon", "obs", "pred"])
        df["test_index"] = self.test_index
        df[["lat", "lon"]] = self.test_coords
        df["obs"] = y_true
        df["pred"] = y_pred

        return rmse, df


if __name__ == "__main__":
    from graph_datasets import SparseObsDataset

    dataset = SparseObsDataset("dataset", "sparse_north", "north")
    data = dataset[4]
    adw = ADW()
    st_trainer = StatisticInterpTrainer(
        adw, data.x, data.y, data.train_index, data.test_index, data.cdist
    )
    print(st_trainer.calculate())
