import time
from contextlib import contextmanager
from typing import List


@contextmanager
def record_time(bucket: List[float], idx: int, start_after: int):
    """
    Time the wrapped block and append its duration (in seconds) to `bucket`.
    Usage:
        with record_time(pre_times):
            ... your block ...
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        if idx >= start_after:
            bucket.append(time.perf_counter() - start)
