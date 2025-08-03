import time
import functools

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"[TIMER] Start running '{func.__name__}'")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[TIMER] '{func.__name__}' completed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper