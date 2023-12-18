import time
from functools import wraps

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