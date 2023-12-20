import numpy as np
from ADW_class import ADW
import torch
import torch.nn as nn
import torch.nn.functional as F
from AGAIN_module import AGAIN
from tqdm.notebook import tqdm
from loguru import logger
import os
from utils import DataFilter, KFoldGnnModel, read_new_zealand_data, seed_everything
import json
import datetime
from copy import deepcopy
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--location", type=str, default="south")
parser.add_argument("--threshold", type=float, default=0.03)
args = parser.parse_args()
threshold = args.threshold
location = args.location
start_month = 1
end_month = 12

seed_everything(2023)

log_path = f'{datetime.datetime.now().strftime("%Y-%m-%d")}-{location}-{threshold}'
os.makedirs(os.path.join(log_path, "comparision_csv"), exist_ok=True)
logger.add(
    f"{log_path}/{location}-{threshold}-{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss(reduction="mean").to(device)
# 和消融实验的配置一致,都是用于降水


log_dict = dict()
if os.path.exists(os.path.join(log_path, "log.json")):
    with open(os.path.join(log_path, "log.json"), "r") as f:
        log_dict = json.load(f)

for file_path in DataFilter("data", start_month, end_month):
    if os.path.exists(
        os.path.join(log_path, "comparision_csv", os.path.basename(file_path) + ".csv")
    ):
        continue
    logger.info(f"Using Data {os.path.basename(file_path)}")
    data = read_new_zealand_data(file_path, location=location)
    kfold = KFoldGnnModel(data, n_splits=len(data), shuffle=True, device=device)
    X, y = kfold.X_tensor, kfold.y_tensor
    loss_df = pd.DataFrame(
        {
            "lat": data["lat"].values,
            "lon": data["lon"].values,
            "obs": data["obs"].values,
        }
    )  # 用于记录每个点对应的loss
    loss_df[["pred", "mse"]] = np.nan  # 预留好位置
    # 记录每折中最小的验证集损失，用于计算平均损失和标准差
    min_test_rmse_loss = []
    min_test_mae_loss = []

    for fold, (train_index, test_index, edge_idx, edge_attr) in enumerate(
        kfold.split()
    ):
        if fold == 50:
            break
        model = AGAIN(
            in_dim=X.shape[1],
            edge_dim=edge_attr.shape[1],
            num_heads=6,
            h1_dim=48,
            h2_dim=60,
            threshold=threshold,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        # ---------记录变量----------------
        test_rmse_loss = []

        for epoch in range(50):
            optimizer.zero_grad()
            y_pred = model(emb_src=X, edge_index=edge_idx, edge_attr=edge_attr)
            loss = criterion(y[train_index], y_pred[train_index])
            # ------------------验证阶段----------------
            if (epoch + 1) % 10 == 0:
                model.eval()
                y_test = model(
                    emb_src=X,
                    edge_index=edge_idx,
                    edge_attr=edge_attr,
                    sparse_train=False,
                )
                loss_test = criterion(y_test[test_index], y[test_index])
                test_rmse_loss.append(loss_test.detach().sqrt().item())  # RMSE
                model.train()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
        # ------------------------一个fold训练结束---------------------------------
        # 就是将测试集预测一个loss 这样根据全部折就能过一遍测试集
        min_test_rmse_loss.append(min(test_rmse_loss))
        # 注意 如果不是留一 需要.numpy()而不是.item()

        loss_df.iloc[test_index, [3, 4]] = [
            torch.clip(y_test[test_index], 0, None).detach().cpu().item(),
            F.mse_loss(
                torch.clip(y_test[test_index], 0, None), y[test_index], reduction="none"
            )
            .detach()
            .cpu()
            .item(),
        ]

        logger.info(
            f"{os.path.basename(file_path)} Fold {fold+1} Test:RMSE:{min(test_rmse_loss):.3f}"
        )
    file_key = os.path.basename(file_path)
    log_dict[file_key] = {
        "test_rmse_loss_mean": np.mean(min_test_rmse_loss),
        "test_rmse_loss_std": np.std(min_test_rmse_loss, ddof=0),
    }
    with open(os.path.join(log_path, "log.json"), "w") as json_file:
        json.dump(log_dict, json_file, indent=4)

    loss_df.dropna(axis=0, inplace=True)  # 去除有缺失值的行
    loss_df.to_csv(
        os.path.join(log_path, "comparision_csv", f"{file_key}.csv"), index=None
    )
