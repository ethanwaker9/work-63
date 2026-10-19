import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import csv
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lgf
from lgf.bench import run

METHODS = [
    ("BDT", lgf.bdt, False),
    ("TFFT", lgf.trapezoid_fft, False),
    ("PFFT", lgf.periodized_fft2, False),
    ("ATR", lgf.adaptive_trapezoid, True),
    ("BQ", lgf.bessel_quadrature, True),
    ("PS", lgf.path_series, True),
    ("HG", lgf.hypergeometric, True),
    ("REC", lgf.marching, False),
]


def feasible(name, c, alpha, L, eps):
    if name == "PFFT":
        g1, g2 = lgf.gammas(c, alpha)
        lc = np.log(16.0 * lgf.decay_constant(c, alpha) / eps)
        P1 = L + 1 + lc / g1
        P2 = L + 1 + lc / g2
        return P1 * P2 * 8 * 1.6 < 2.5e9
    if name == "PS":
        K = lgf.series_terms(c, alpha, eps)
        return (L + 1) ** 2 * max(K - L, 1) ** 2 / 8 < 3e8
    if name == "HG":
        return L <= 31 and c >= 0.3
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--part", choices=["all", "sizes", "small", "sweep"], default="all")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    eps = args.eps
    sizes = [63, 127, 255, 511, 1023, 2047]
    cases = [(0.01, 1.0), (0.1, 1.0), (1.0, 1.0), (0.05, 0.25), (0.05, 4.0)]
    if args.quick:
        sizes = [3, 7, 15, 31, 63]
        cases = [(0.1, 1.0), (1.0, 1.0)]
    rows = []
    path = os.path.join(args.out, "benchmark_sizes.csv")
    for (c, alpha) in (cases if args.part in ("all", "sizes") else []):
        stopped = set()
        for L in sizes:
            R = lgf.reference_table(c, alpha, L, L)
            for name, fun, slow in METHODS:
                if name in stopped or not feasible(name, c, alpha, L, eps):
                    continue
                out, t, peak = run(fun, (c, alpha, L, L, eps), slow=slow)
                if name == "BDT":
                    out = lgf.expand(out, L, L)
                err = float(np.abs(out - R).max())
                rows.append(dict(c=c, alpha=alpha, L=L, method=name, time=t, peak_bytes=peak,
                                 table_bytes=8 * (L + 1) ** 2, max_abs_err=err))
                print(f"c={c} alpha={alpha} L={L} {name}: {t:.4g}s {peak / 1e6:.4g}MB err={err:.2e}", flush=True)
                if t * 4.5 > args.cap:
                    stopped.add(name)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    small = []
    for c in ([1.0] if args.part in ("all", "small") else []):
        L = 31
        R = lgf.reference_table(c, 1.0, L, L)
        for name, fun, slow in METHODS:
            out, t, peak = run(fun, (c, 1.0, L, L, eps), slow=slow)
            if name == "BDT":
                out = lgf.expand(out, L, L)
            err = float(np.abs(out - R).max())
            small.append(dict(c=c, alpha=1.0, L=L, method=name, time=t, peak_bytes=peak,
                              table_bytes=8 * (L + 1) ** 2, max_abs_err=err))
            print(f"small c={c} L={L} {name}: {t:.4g}s {peak / 1e6:.4g}MB err={err:.2e}", flush=True)
    if small:
        with open(os.path.join(args.out, "benchmark_small.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(small[0].keys()))
            w.writeheader()
            w.writerows(small)
    sweep = []
    cs = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    if args.quick:
        cs = [0.1, 1.0]
    L = 255 if not args.quick else 31
    for c in (cs if args.part in ("all", "sweep") else []):
        R = lgf.reference_table(c, 1.0, L, L)
        for name, fun, slow in METHODS:
            if name in ("REC", "ATR", "BQ", "PS", "HG"):
                continue
            if not feasible(name, c, 1.0, L, eps):
                continue
            out, t, peak = run(fun, (c, 1.0, L, L, eps), slow=slow)
            if name == "BDT":
                out = lgf.expand(out, L, L)
            err = float(np.abs(out - R).max())
            sweep.append(dict(c=c, alpha=1.0, L=L, method=name, time=t, peak_bytes=peak,
                              table_bytes=8 * (L + 1) ** 2, max_abs_err=err))
            print(f"sweep L={L} c={c} {name}: {t:.4g}s {peak / 1e6:.4g}MB err={err:.2e}", flush=True)
        with open(os.path.join(args.out, "benchmark_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            w.writeheader()
            w.writerows(sweep)


if __name__ == "__main__":
    main()
