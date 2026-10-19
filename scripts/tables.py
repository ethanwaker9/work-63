import os
import csv
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
ORDER = ["BDT", "TFFT", "PFFT", "ATR", "BQ", "PS", "HG", "REC"]


def read(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def fmt_time(t):
    t = float(t)
    if t >= 100:
        return f"{t:.0f}"
    if t >= 1:
        return f"{t:.1f}"
    e = int(np.floor(np.log10(t)))
    m = t / 10 ** e
    return f"${m:.1f}\\cdot10^{{{e}}}$"


def fmt_mem(b):
    b = float(b) / 1e6
    if not np.isfinite(b):
        return "--"
    if b >= 100:
        return f"{b:.0f}"
    if b >= 10:
        return f"{b:.1f}"
    if b >= 1:
        return f"{b:.1f}"
    return f"{b:.3f}"


def fmt_err(e):
    e = float(e)
    if not np.isfinite(e):
        return "unstable"
    if e > 1.0:
        x = int(np.floor(np.log10(e)))
        return f"$10^{{{x}}}$"
    x = int(np.floor(np.log10(max(e, 1e-300))))
    m = e / 10 ** x
    return f"${m:.0f}\\cdot10^{{{x}}}$"


def nodes_table():
    rows = read("quadrature_nodes.csv")
    cs = sorted({float(r["c"]) for r in rows})
    lines = ["\\begin{tabular}{rrrrrrr}", "\\toprule",
             " & \\multicolumn{3}{c}{$\\eps=10^{-14}$} & \\multicolumn{3}{c}{$\\eps=10^{-8}$}\\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "$c$ & $N_{\\rm opt}$ & ours & exact & $N_{\\rm opt}$ & ours & exact\\\\", "\\midrule"]
    for c in cs:
        a = [r for r in rows if float(r["c"]) == c and float(r["eps"]) == 1e-14][0]
        b = [r for r in rows if float(r["c"]) == c and float(r["eps"]) == 1e-8][0]
        lines.append(f"{c:g} & {a['hc_opt']} & {a['ours']} & {a['exact_min']} & {b['hc_opt']} & {b['ours']} & {b['exact_min']}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def lookup(rows, name, c, alpha, L):
    for x in rows:
        if x["method"] == name and float(x["c"]) == c and float(x["alpha"]) == alpha and int(x["L"]) == L:
            return x
    return None


def bench_table():
    rows = read("benchmark_sizes.csv")
    small = read("benchmark_small.csv")
    sweep = read("benchmark_sweep.csv")
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             " & \\multicolumn{3}{c}{time (s)} & mem (MB) & \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}",
             "method & $c{=}1$, $L{=}31$ & $c{=}0.01$, $L{=}127$ & $c{=}0.01$, $L{=}2047$ & $c{=}0.01$, $L{=}2047$ & max error\\\\",
             "\\midrule"]
    for name in ORDER:
        a = lookup(small, name, 1.0, 1.0, 31)
        b = lookup(rows, name, 0.01, 1.0, 127)
        d = lookup(rows, name, 0.01, 1.0, 2047)
        cells = [fmt_time(a["time"]) if a else "--", fmt_time(b["time"]) if b else "--",
                 fmt_time(d["time"]) if d else "--", fmt_mem(d["peak_bytes"]) if d else "--"]
        errs = [float(x["max_abs_err"]) for x in rows + small + sweep if x["method"] == name]
        err = max(errs) if errs else float("nan")
        if any(not np.isfinite(e) for e in errs):
            err = float("nan")
        label = "\\textbf{BDT (ours)}" if name == "BDT" else name
        lines.append(label + " & " + " & ".join(cells) + " & " + fmt_err(err) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def speedups():
    rows = read("benchmark_sizes.csv")
    out = {}
    for r in rows:
        key = (float(r["c"]), float(r["alpha"]), int(r["L"]))
        out.setdefault(key, {})[r["method"]] = (float(r["time"]), float(r["peak_bytes"]), float(r["max_abs_err"]))
    best = []
    for key, d in out.items():
        if "BDT" not in d:
            continue
        tb = d["BDT"][0]
        for m, (t, p, e) in d.items():
            if m in ("BDT", "REC"):
                continue
            best.append((t / tb, m, key))
    best.sort()
    return best


def main():
    with open(os.path.join(RES, "table_nodes.tex"), "w") as f:
        f.write(nodes_table())
    if os.path.exists(os.path.join(RES, "benchmark_sizes.csv")):
        with open(os.path.join(RES, "table_benchmark.tex"), "w") as f:
            f.write(bench_table())
        sp = speedups()
        with open(os.path.join(RES, "speedups.txt"), "w") as f:
            for s in sp:
                f.write(f"{s[0]:.3g} {s[1]} {s[2]}\n")


if __name__ == "__main__":
    main()
