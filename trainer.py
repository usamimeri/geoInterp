from typing import Literal
import torch
from models.interpolators import ADW, AGAIN, GATModel, GCNModel
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from graph_datasets import SparseObsDataset
from moran import LocalMoranIndex
import os
import json


class StatisticInterpTrainer:
    """
    传统统计方法类的推理，由于是统计方法没有学习率之类训练参数。
    """

    def __init__(self, model) -> None:
        self.model = model

    def calculate(self, data: Data):
        """
        计算模型在测试集上的性能。
        参数:
            data (Data): 包含节点特征、标签和索引的PyTorch Geometric Data对象。
        返回:
            tuple: 包含RMSE和预测结果的dataframe。
        """
        y_true, y_pred = self._get_true_and_pred_values(data.cpu())
        rmse = self._calculate_rmse(y_true, y_pred)
        df = self._create_dataframe(data.cpu(), y_true, y_pred)

        return rmse, df

    def _get_true_and_pred_values(self, data: Data):
        """
        从数据中获取真实值和预测值。
        """
        y_true = data.y[data.test_index].numpy()
        y_pred = self.model.interpolate(
            data.x[data.train_index, 0:2].numpy(),
            data.x[data.test_index, 0:2].numpy(),
            data.y[data.train_index].numpy(),
            data.cdist.numpy(),
        )
        return y_true, y_pred

    def _calculate_rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_pred - y_true) ** 2))

    def _create_dataframe(self, data: Data, y_true, y_pred):
        """
        创建包含测试集索引、坐标、观测值和预测值的dataframe。
        """
        df = pd.DataFrame(
            {
                "test_index": data.test_index,
                "lat": data.x[data.test_index, 0].numpy(),
                "lon": data.x[data.test_index, 1].numpy(),
                "obs": y_true,
                "pred": y_pred,
            }
        )

        return df


class GNNTrainer:
    def __init__(
        self,
        model_type: Literal["GAT", "GCN", "AGAIN"],
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_epochs=100,
    ) -> None:
        self.model_type = model_type
        self.criterion = torch.nn.MSELoss().to(device)
        self.max_epochs = max_epochs
        self.device = device
        self.moran = LocalMoranIndex()

    def calculate(self, data: Data):
        if self.model_type == "GAT":
            model = GATModel(data.x.shape[1])
        elif self.model_type == "GCN":
            model = GCNModel(data.x.shape[1])
        elif self.model_type == "AGAIN":
            model = AGAIN(
                in_dim=data.x.shape[1],
                edge_dim=data.edge_attr.shape[1],
                num_heads=6,
                h1_dim=48,
                h2_dim=60,
                threshold=0.03,
            )
        else:
            raise NotImplementedError

        best_rmse_loss = 1e9
        best_y_pred = None

        model = model.to(self.device)
        data = data.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=5e-4)

        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            y_pred = model(data.x, data.edge_index, data.edge_attr)

            if self.model_type == "AGAIN":
                moran = self.moran(
                    y_pred[data.train_index] - data.y[data.train_index],
                    data.train_weights,
                    data.train_neighbors,
                )
                loss = (
                    self.criterion(data.y[data.train_index], y_pred[data.train_index])
                    + 0.05 * moran
                )
            else:
                loss = self.criterion(
                    data.y[data.train_index], y_pred[data.train_index]
                )

            if (epoch + 1) % 20 == 0:
                model.eval()
                y_test = model(
                    data.x,
                    data.edge_index,
                    data.edge_attr,
                    False,  # Sparse Train,Only available to AGAIN
                )
                y_test = model(data.x, data.edge_index, data.edge_attr)
                loss_test = (
                    self.criterion(data.y[data.test_index], y_test[data.test_index])
                    .detach()
                    .sqrt()
                    .item()
                )
                if best_rmse_loss > loss_test:
                    best_rmse_loss = loss_test
                    best_y_pred = y_test[data.test_index]
                model.train()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

        df = self._create_dataframe(data, best_y_pred.detach().cpu().numpy())
        return best_rmse_loss, df

    def _create_dataframe(self, data: Data, y_pred):
        """
        创建包含测试集索引、坐标、观测值和预测值的dataframe。
        """
        data = data.cpu()
        df = pd.DataFrame(
            {
                "test_index": data.test_index,
                "lat": data.x[data.test_index, 0].numpy(),
                "lon": data.x[data.test_index, 1].numpy(),
                "obs": data.y[data.test_index].numpy(),
                "pred": y_pred,
            }
        )

        return df


if __name__ == "__main__":
    location = "south"
    # dataset = SparseObsDataset("dataset", "sparse_north", "north")
    dataset = SparseObsDataset("dataset", f"sparse_{location}", location)
    print("Initializing……")
    gat_trainer = GNNTrainer("GAT")
    gcn_trainer = GNNTrainer("GCN")
    again_trainer = GNNTrainer("AGAIN")
    adw_trainer = StatisticInterpTrainer(ADW())
    trainers = {
        "GAT": gat_trainer,
        "GCN": gcn_trainer,
        "AGAIN": again_trainer,
        "ADW": adw_trainer,
    }

    logs = {
        "GAT": {"10%": [], "20%": [], "30%": [], "40%": [], "50%": []},
        "GCN": {"10%": [], "20%": [], "30%": [], "40%": [], "50%": []},
        "AGAIN": {"10%": [], "20%": [], "30%": [], "40%": [], "50%": []},
        "ADW": {"10%": [], "20%": [], "30%": [], "40%": [], "50%": []},
    }

    device = torch.device("cuda")

    sparsities = ["10%", "20%", "30%", "40%", "50%"] * len(dataset.raw_file_names)
    output_dir = f"outputs_{location}_sparse"
    for model_name in list(trainers.keys()):
        for sparsity in sparsities[:5]:
            os.makedirs(
                os.path.join(output_dir, model_name, sparsity),
                exist_ok=True,
            )
    print("Done!")
    for i in range(len(dataset.processed_file_names)):
        data = dataset[i].to(device)
        print("Using ", dataset.processed_file_names[i])
        for model_name in list(trainers.keys()):
            trainer = trainers[model_name]
            rmse, df = trainer.calculate(data)
            print(model_name, rmse)
            logs[model_name][sparsities[i]].append(rmse)
            df.to_csv(
                os.path.join(
                    output_dir,
                    model_name,
                    sparsities[i],
                    f"{dataset.processed_file_names[i].split('.')[0]}.csv",
                ),
                index=False,
            )
        if i % 20 == 0:
            with open(os.path.join(output_dir, "logs.json"), "w") as f:
                json.dump(logs, f)
