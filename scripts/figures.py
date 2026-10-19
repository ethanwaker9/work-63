import os
import sys
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lgf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "axes.linewidth": 0.6,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})

STYLE = {
    "BDT": dict(color="#c0392b", marker="o", ls="-", lw=1.6, label="BDT (ours)"),
    "TFFT": dict(color="#1f77b4", marker="s", ls="-", label="TFFT"),
    "PFFT": dict(color="#2ca02c", marker="^", ls="-", label="PFFT"),
    "ATR": dict(color="#9467bd", marker="v", ls="-", label="ATR"),
    "BQ": dict(color="#8c564b", marker="D", ls="-", label="BQ"),
    "PS": dict(color="#e377c2", marker="P", ls="-", label="PS"),
    "HG": dict(color="#7f7f7f", marker="X", ls="-", label="HG"),
    "REC": dict(color="#bcbd22", marker="*", ls="--", label="REC"),
}


def read(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def fig_decay():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 1.75))
    ax = axs[0]
    cols = ["#1f77b4", "#2ca02c", "#c0392b"]
    for c, col in zip([0.05, 0.2, 1.0], cols):
        alpha = 1.0
        nmax = {0.05: 400, 0.2: 200, 1.0: 60}[c]
        ns = np.unique(np.linspace(0, nmax, 26).astype(int))
        B = np.array([lgf.reference(n, n, c, alpha) for n in ns])
        ours = np.array([np.exp(-lgf.kappa(n, n, c, alpha)) * lgf.decay_constant(c, alpha) for n in ns])
        lam = 2.0 + 2.0 * alpha
        hc = (lam / (lam + c * c)) ** (2 * ns) / (c * c)
        ax.semilogy(ns, B, "o", color=col, mfc="none", ms=3.2, label=f"$B_c(n,n)$, $c={c}$")
        ax.semilogy(ns, ours, "-", color=col, lw=1.1)
        ax.semilogy(ns, hc, ":", color=col, lw=1.1)
    ax.plot([], [], "k-", lw=1.1, label="Theorem 3.4")
    ax.plot([], [], "k:", lw=1.1, label="bound of Hou and Colonius")
    ax.set_xlabel("lattice index $n$ on the diagonal $(n,n)$")
    ax.set_ylabel("value")
    ax.set_ylim(1e-80, 1e6)
    ax.legend(loc="lower right", ncol=1, frameon=False, handlelength=1.6, fontsize=5.8, borderaxespad=0.2)
    ax.set_title("(a) decay along the diagonal")
    ax = axs[1]
    rows = read("expansion_error.csv")
    sel = [r for r in rows if float(r["c"]) == 0.5 and float(r["alpha"]) == 0.5 and float(r["nu"]) == 0.5]
    for J, col in zip([1, 2, 3, 4, 5], ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#c0392b"]):
        s = [r for r in sel if int(r["J"]) == J]
        ms = np.array([float(r["m"]) for r in s])
        es = np.array([float(r["rel_err"]) for r in s])
        ax.loglog(ms, es, "o-", color=col, label=f"$J={J}$")
        ax.loglog(ms, es[-1] * (ms / ms[-1]) ** (-J), ":", color=col, lw=0.8)
    ax.set_xlabel("$m$ (with $n=m/2$)")
    ax.set_ylabel("relative error")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    ax.set_title("(b) far field expansion, $c=0.5$, $\\alpha=0.5$")
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(os.path.join(FIG, "decay_expansion.eps"))
    plt.close(fig)


def fig_bench():
    rows = read("benchmark_sizes.csv")
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 1.7))
    for ax, c, key, ttl in [(axs[0], 0.01, "time", "(a) time, $c=0.01$"),
                            (axs[1], 0.01, "peak_bytes", "(b) peak memory, $c=0.01$"),
                            (axs[2], 1.0, "time", "(c) time, $c=1$")]:
        for name, st in STYLE.items():
            s = [r for r in rows if r["method"] == name and float(r["c"]) == c and float(r["alpha"]) == 1.0]
            if not s:
                continue
            L = np.array([int(r["L"]) + 1 for r in s])
            v = np.array([float(r[key]) for r in s])
            if key == "peak_bytes":
                v = v / 1e6
            ax.loglog(L, v, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st.get("lw", 1.0),
                      label=st["label"], mfc="none" if name != "BDT" else st["color"])
        if key == "peak_bytes":
            Ls = np.array([50, 2200])
            ax.loglog(Ls, 8.0 * Ls ** 2 / 1e6, "k:", lw=0.8, label="table size")
            ax.set_ylabel("peak memory (MB)")
            ax.set_xlim(45, 2600)
            ax.set_ylim(1e-6, 3e3)
        else:
            ax.set_ylabel("wall time (s)")
        ax.set_xlabel("table side $L+1$")
        ax.set_title(ttl)
    hs, ls = [], []
    for ax in axs:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls:
                hs.append(h)
                ls.append(l)
    axs[1].legend(hs, ls, loc="lower right", bbox_to_anchor=(1.02, -0.02), frameon=False, ncol=2, handlelength=1.4,
                  columnspacing=0.5, fontsize=5.8, borderaxespad=0.2)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(os.path.join(FIG, "benchmark.eps"))
    plt.close(fig)


def fig_sweep():
    rows = read("benchmark_sweep.csv")
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.3))
    for name, st in STYLE.items():
        s = [r for r in rows if r["method"] == name and int(r["L"]) == 255]
        if not s:
            continue
        cs = np.array([float(r["c"]) for r in s])
        t = np.array([float(r["time"]) for r in s])
        ax.loglog(cs, t, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st.get("lw", 1.0),
                  label=st["label"], mfc="none" if name != "BDT" else st["color"])
    ax.set_xlabel("screening $c$")
    ax.set_ylabel("wall time (s)")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "sweep.eps"))
    plt.close(fig)


def fig_surface():
    c, alpha = 0.5, 0.5
    L = 30
    T = lgf.reference_table(c, alpha, L, L)
    Tf = np.concatenate([T[:0:-1], T], axis=0)
    Tf = np.concatenate([Tf[:, :0:-1], Tf], axis=1)
    n = np.arange(-L, L + 1)
    Nn, Mm = np.meshgrid(n, n, indexing="ij")
    Z = np.log10(Tf)
    K = lgf.kappa_grid(Nn, Mm, c, alpha)
    W = np.log10(lgf.decay_constant(c, alpha)) - K / np.log(10.0)
    fig = plt.figure(figsize=(3.35, 2.25))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Nn, Mm, Z, cmap="viridis", rstride=2, cstride=2, linewidth=0.1, edgecolor="none", antialiased=False)
    ax.plot_wireframe(Nn, Mm, W, rstride=6, cstride=6, color="#c0392b", linewidth=0.45)
    ax.set_zlim(-17.0, 1.0)
    ax.set_xlabel("$n$", labelpad=-7)
    ax.set_ylabel("$m$", labelpad=-7)
    ax.set_zlabel("$\\log_{10}$ value", labelpad=-6)
    ax.tick_params(axis="x", pad=-4, labelsize=6)
    ax.tick_params(axis="y", pad=-4, labelsize=6)
    ax.tick_params(axis="z", pad=-2, labelsize=6)
    ax.set_xticks([-30, 0, 30])
    ax.set_yticks([-30, 0, 30])
    ax.set_zticks([-15, -10, -5, 0])
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((1, 1, 0.7))
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.03, top=1.0)
    fig.savefig(os.path.join(FIG, "surface3d.eps"), dpi=300)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_decay()
    fig_surface()
    if os.path.exists(os.path.join(RES, "benchmark_sizes.csv")):
        fig_bench()
    if os.path.exists(os.path.join(RES, "benchmark_sweep.csv")):
        fig_sweep()


if __name__ == "__main__":
    main()
