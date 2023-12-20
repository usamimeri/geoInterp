import time
from functools import wraps
from haversine import haversine_vector, Unit
from scipy.spatial.distance import cdist
import datetime
import os
from typing import List, Literal, Union
import pandas as pd
import random
import numpy as np
import torch


class DataFilter:
    """
    根据指定时间范围选取年份下的数据
    例如选取12-20 到 12-22 就会获取data文件夹下所有年份的对应数据集

    随后可以使用迭代的方式来获取对应的数据路径
    """

    def __init__(
        self,
        root: str,
        start_month: int,
        end_month: int,
    ) -> List[str]:
        self.root = root
        self.start_month = start_month
        self.end_month = end_month
        self.valid_file_names = []
        self.current_index = 0
        self.get_valid_file_name()

    def parse_date_from_filename(self, filename):
        # 假设文件名的前8个字符是日期（格式：yyyyMMdd）
        return datetime.datetime.strptime(filename[:8], "%Y%m%d")

    def is_date_within_range(
        self,
        file_date: datetime.datetime,
    ) -> bool:
        # 判断是否在指定的月份范围内，不关心哪一年
        return self.start_month <= file_date.month <= self.end_month

    def get_valid_file_name(self):
        for root, dirs, files in os.walk(self.root):
            if root == self.root:
                continue
            for file in files:
                file_date = self.parse_date_from_filename(file)
                if self.is_date_within_range(file_date):
                    self.valid_file_names.append(os.path.join(root, file))

    def __iter__(self):
        return self

    def __next__(self):
        """
        用迭代器的方式依次读取文件名，以便放入到后续的文件读取中
        """
        if self.current_index < len(self.valid_file_names):
            result = self.valid_file_names[self.current_index]
            self.current_index += 1
            return result
        else:
            raise StopIteration


def measure_runtime(repetitions=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            total_time = 0
            for _ in range(repetitions):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                total_time += end_time - start_time
            average_time = total_time / repetitions
            print(
                f"{func.__name__} took an average of {average_time:.4f} seconds to run over {repetitions} repetitions."
            )
            return result

        return wrapper

    return decorator


def cal_haversine_dists_matrix(X, Y=None, unit=Unit.KILOMETERS):
    """
    计算地理坐标点集之间的距离矩阵。使用哈弗辛公式（Haversine formula）来计算两点之间的大圆距离。

    Parameters:
    X (numpy.ndarray): 形状为 (N, 2)，其中 N 是点的数量。每个点的格式应为 (纬度, 经度)。
    Y (numpy.ndarray, optional): 形状为 (M, 2)。若未提供，将使用 X 计算内部成对距离。默认为 None。
    unit (Unit, optional): 距离单位，可以是 Unit.KILOMETERS、Unit.MILES 等。默认为 Unit.KILOMETERS。

    Returns:
    numpy.ndarray: 两个点集之间的距离矩阵。如果 Y 被提供，返回形状为 (N, M) 的矩阵；
    如果 Y 未提供，返回形状为 (N, N) 的矩阵，表示 X 中所有点对的距离。

    Example:
    >>> X = np.array([[39.93, 116.40], [38.89, 77.01]])
    >>> Y = np.array([[37.77, 122.41], [34.05, 118.24]])
    >>> dist_matrix = cal_dists_matrix(X, Y)
    >>> print(dist_matrix)
    """
    if Y is not None:
        return haversine_vector(X, Y, comb=True, unit=unit).T
    else:
        return haversine_vector(X, X, comb=True, unit=unit)


def cal_cos_dists_matrix(X, Y=None, origin=None):
    """
    origin:(2,) 参考原点，会被广播到(M,2)或者(N,2)
    """

    if origin is not None:
        X = X - origin + 1e-10  # 避免除数为0
    if Y is not None:
        raise NotImplementedError

    # 用于配对矩阵的广播运算
    return cdist(X, X, metric="cosine")


class DataFilter:
    """
    根据指定时间范围选取年份下的数据
    例如选取12-20 到 12-22 就会获取data文件夹下所有年份的对应数据集

    随后可以使用迭代的方式来获取对应的数据路径
    """

    def __init__(
        self,
        root: str,
        start_month: int,
        end_month: int,
    ) -> List[str]:
        self.root = root
        self.start_month = start_month
        self.end_month = end_month
        self.valid_file_names = []
        self.current_index = 0
        self.get_valid_file_name()

    def parse_date_from_filename(self, filename):
        # 假设文件名的前8个字符是日期（格式：yyyyMMdd）
        return datetime.datetime.strptime(filename[:8], "%Y%m%d")

    def is_date_within_range(
        self,
        file_date: datetime.datetime,
    ) -> bool:
        # 判断是否在指定的月份范围内，不关心哪一年
        return self.start_month <= file_date.month <= self.end_month

    def get_valid_file_name(self):
        for root, dirs, files in os.walk(self.root):
            if root == self.root:
                continue
            for file in files:
                file_date = self.parse_date_from_filename(file)
                if self.is_date_within_range(file_date):
                    self.valid_file_names.append(os.path.join(root, file))

    def __iter__(self):
        return self

    def __next__(self):
        """
        用迭代器的方式依次读取文件名，以便放入到后续的文件读取中
        """
        if self.current_index < len(self.valid_file_names):
            result = self.valid_file_names[self.current_index]
            self.current_index += 1
            return result
        else:
            raise StopIteration


def get_location_mask(lat: Union[float, List[float]], lon: Union[float, List[float]]):
    """
    用于新西兰数据生成南北半岛标签.
    True - 给定坐标位于新西兰南半岛; False - 给定坐标位于新西兰北半岛
    """

    p1, p2 = (174.6, -41.3), (174.43, -41.1)
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - k * p1[0]

    if type(lat) == float:
        return True if -k * lon + lat <= b else False
    else:
        return np.array(
            [
                True if -k * lontitude + latitude <= b else False
                for latitude, lontitude in zip(lat, lon)
            ]
        )


def read_new_zealand_data(
    file_path: str, location: Literal["north", "south", "all"]
) -> pd.DataFrame:
    """
    读取新西兰数据集，数据集以.dat结尾
    一般列是：
    * index：站点标签，可忽略
    * lat：纬度
    * lon：经度
    * elev：高程
    * obs：观测值
    * CMORPH：红外数据
    """
    data = pd.read_csv(
        file_path,
        header=None,
        names=["index", "lat", "lon", "elev", "obs", "CMORPH"],
        sep="\s+",
        usecols=["lat", "lon", "elev", "obs"],
    )  # 空格长度不太一致

    if location == "all":
        return data
    elif location == "south":
        mask = get_location_mask(data["lat"].values, data["lon"].values)
        return data[mask]
    elif location == "north":
        mask = get_location_mask(data["lat"].values, data["lon"].values)
        mask = ~mask  # 取反
        return data[mask]
    else:
        raise ValueError


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    print(cal_cos_dists_matrix(np.array([[10, 20], [90, 0], [0, 90]])))
