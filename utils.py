import time
from functools import wraps
from haversine import haversine_vector, Unit

def measure_runtime(repetitions=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            total_time = 0
            for _ in range(repetitions):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                total_time += (end_time - start_time)
            average_time = total_time / repetitions
            print(f"{func.__name__} took an average of {average_time:.4f} seconds to run over {repetitions} repetitions.")
            return result
        return wrapper
    return decorator

def cal_dists_matrix(X, Y=None, unit=Unit.KILOMETERS):
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