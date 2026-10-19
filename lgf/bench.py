import gc
import time
import tracemalloc
import numpy as np


def peak_memory(fun, args):
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = fun(*args)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return out, peak


def best_time(fun, args, min_total=0.3, max_repeats=15, first=None):
    times = [] if first is None else [first]
    total = sum(times)
    while len(times) < max_repeats and (total < min_total or len(times) < 3):
        gc.collect()
        t0 = time.perf_counter()
        fun(*args)
        dt = time.perf_counter() - t0
        times.append(dt)
        total += dt
        if dt > 2.0:
            break
    return min(times)


def run(fun, args, slow=False):
    gc.collect()
    t0 = time.perf_counter()
    fun(*args)
    first = time.perf_counter() - t0
    if slow or first > 2.0:
        t = first
    else:
        t = best_time(fun, args, first=first)
    out, peak = peak_memory(fun, args) if first < 60.0 else (fun(*args), np.nan)
    return out, t, peak
