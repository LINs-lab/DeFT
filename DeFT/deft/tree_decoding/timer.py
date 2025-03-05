import time
import torch


class GlobalTimer:
    started_timer: dict[str, float] = {}
    total_timer: dict[str, float] = {}

    def __init__(self) -> None:
        pass

    @staticmethod
    def start(key: str) -> None:
        if key not in GlobalTimer.started_timer:
            GlobalTimer.started_timer[key] = 0
        torch.cuda.synchronize()
        # torch.cuda.nvtx.range_push(key)
        GlobalTimer.started_timer[key] = time.perf_counter()

    @staticmethod
    def stop(key: str) -> float:
        if key not in GlobalTimer.total_timer:
            GlobalTimer.total_timer[key] = 0
        torch.cuda.synchronize()
        # torch.cuda.nvtx.range_pop()
        t = (time.perf_counter() - GlobalTimer.started_timer[key]) * 1000
        GlobalTimer.total_timer[key] += t
        return t

    @staticmethod
    def reset(key: str) -> None:
        GlobalTimer.total_timer[key] = 0

    @staticmethod
    def get(key: str) -> float:
        return GlobalTimer.total_timer[key]
