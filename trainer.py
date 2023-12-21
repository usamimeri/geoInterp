from typing import Literal
import torch
from models.interpolators import ADW, AGAIN, GATModel, GCNModel, KCN
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
        model_type: Literal["GAT", "GCN", "KCN", "AGAIN"],
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
        elif self.model_type == "KCN":
            model = KCN(10, device, data)
        else:
            raise NotImplementedError

        best_rmse_loss = 1e9
        best_y_pred = None

        model = model.to(self.device)
        data = data.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)

        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            if self.model_type == "KCN":
                y_pred = model(data.train_index)
            else:
                y_pred = model(data.x, data.edge_index, data.edge_attr)

            if self.model_type == "AGAIN":
                # moran = self.moran(
                #     y_pred[data.train_index] - data.y[data.train_index],
                #     data.train_weights,
                #     data.train_neighbors,
                # )
                loss = self.criterion(
                    data.y[data.train_index], y_pred[data.train_index]
                )
                # + 0.05 * moran
            elif self.model_type == "KCN":
                loss = self.criterion(data.y[data.train_index].reshape(-1, 1), y_pred)
            else:
                loss = self.criterion(
                    data.y[data.train_index], y_pred[data.train_index]
                )

            if (epoch + 1) % 10 == 0:
                model.eval()
                """
                需要改进这部分代码
                """
                if self.model_type == "KCN":
                    y_test = model(data.test_index)
                    loss_test = (
                        self.criterion(data.y[data.test_index].reshape(-1, 1), y_test)
                        .detach()
                        .sqrt()
                        .item()
                    )
                else:
                    y_test = model(
                        data.x,
                        data.edge_index,
                        data.edge_attr,
                        False,  # Sparse Train,Only available to AGAIN
                    )
                    loss_test = (
                        self.criterion(data.y[data.test_index], y_test[data.test_index])
                        .detach()
                        .sqrt()
                        .item()
                    )
                if best_rmse_loss > loss_test:
                    best_rmse_loss = loss_test
                    if self.model_type == "KCN":
                        best_y_pred = y_test
                    else:
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


from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--model_type", type=str, choices=["GAT", "GCN", "AGAIN", "ADW"])
parser.add_argument("--max_epochs", type=int, default=150)
parser.add_argument("--location", type=str)
parser.add_argument("--log_step", type=int, default=50)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Initializing……")
    dataset = SparseObsDataset("dataset", f"sparse_{args.location}", args.location)

    trainer = (
        GNNTrainer(args.model_type, max_epochs=args.max_epochs)
        if args.model_type in ["GCN", "GAT", "AGAIN"]
        else StatisticInterpTrainer(ADW(n_neighbors=20, cdd=60, m=4))
    )

    logs = {
        args.model_type: {"10%": [], "20%": [], "30%": [], "40%": [], "50%": []},
    }

    device = torch.device("cuda")

    sparsities = ["10%", "20%", "30%", "40%", "50%"] * len(dataset.raw_file_names)
    output_dir = f"outputs_{args.location}_sparse"
    for sparsity in sparsities[:5]:
        os.makedirs(
            os.path.join(output_dir, args.model_type, sparsity),
            exist_ok=True,
        )
    print("Done!")

    for i in tqdm(range(len(dataset.processed_file_names)), desc=args.model_type):
        data = dataset[i].to(device)
        rmse, df = trainer.calculate(data)
        logs[args.model_type][sparsities[i]].append(rmse)
        df.to_csv(
            os.path.join(
                output_dir,
                args.model_type,
                sparsities[i],
                f"{dataset.processed_file_names[i].split('.')[0]}.csv",
            ),
            index=False,
        )
        if (i % args.log_step) == 0:
            with open(os.path.join(output_dir, args.model_type, "logs.json"), "w") as f:
                json.dump(logs, f, indent=4)

    with open(os.path.join(output_dir, args.model_type, "logs.json"), "w") as f:
        json.dump(logs, f, indent=4)
