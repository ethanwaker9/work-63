import os
import sys
import csv
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lgf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

HC_TABLE = {
    1e-14: {0.001: (41510, 41518), 0.005: (7977, 7979), 0.01: (3918, 3920), 0.05: (752, 752), 0.1: (369, 369), 0.5: (72, 72), 1.0: (36, 36)},
    1e-11: {0.001: (34511, 34541), 0.005: (6576, 6583), 0.01: (3218, 3222), 0.05: (611, 612), 0.1: (299, 300), 0.5: (57, 58), 1.0: (29, 29)},
    1e-8: {0.001: (27490, 27563), 0.005: (5171, 5188), 0.01: (2515, 2524), 0.05: (471, 473), 0.1: (229, 230), 0.5: (43, 43), 1.0: (22, 22)},
}


def write(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def check_decay():
    rows = []
    worst = 0.0
    for c in [0.005, 0.02, 0.1, 0.5, 1.0, 3.0]:
        for alpha in [0.2, 0.5, 1.0, 2.0, 5.0]:
            for m in [0, 1, 4, 16, 64, 200]:
                for n in [0, 1, 3, 9, 27, 81, 200]:
                    B = lgf.reference(n, m, c, alpha)
                    k = lgf.kappa(n, m, c, alpha)
                    if not (B > 0.0) or k > 650.0:
                        continue
                    ratio = float(np.exp(np.log(B) + k - np.log(lgf.decay_constant(c, alpha))))
                    worst = max(worst, ratio)
                    rows.append(dict(c=c, alpha=alpha, n=n, m=m, B=B, kappa=k, ratio=ratio,
                                     r=float(np.hypot(n, m))))
    write("decay_ratio.csv", rows)
    return worst


def check_aliasing():
    rows = []
    worst = 0.0
    for c, alpha in [(0.05, 1.0), (0.3, 0.5), (1.0, 2.0)]:
        g1, _ = lgf.gammas(c, alpha)
        for N in [64, 128, 256]:
            T = lgf.trap_row(3, c, alpha, N)
            for n in [0, 5, 20, N // 2 - 1]:
                lmax = int(np.ceil(45.0 / (g1 * N))) + 1
                alias = sum(lgf.reference(n + l * N, 3, c, alpha) for l in range(-lmax, lmax + 1) if l != 0)
                B = lgf.reference(n, 3, c, alpha)
                d = abs((T[n] - B) - alias)
                worst = max(worst, d / T[0])
                rows.append(dict(c=c, alpha=alpha, N=N, n=n, trap_minus_B=T[n] - B, alias_sum=alias, diff=d))
    write("aliasing_identity.csv", rows)
    return worst


def minimal_nodes(c, alpha, eps):
    g1, _ = lgf.gammas(c, alpha)
    C = lgf.decay_constant(c, alpha)
    hi = int(np.ceil(np.log1p(2.0 * C / eps) / g1))
    err = lambda N: 2.0 * sum(lgf.reference(l * N, 0, c, alpha) for l in range(1, 4))
    lo = max(2, int(0.5 * hi))
    while err(lo) <= eps:
        lo = max(2, lo // 2)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if err(mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi


def check_nodes():
    rows = []
    for eps, tab in HC_TABLE.items():
        for c, (nopt, nap) in tab.items():
            g1, _ = lgf.gammas(c, 1.0)
            C = lgf.decay_constant(c, 1.0)
            nsuf = int(np.ceil(np.log1p(2.0 * C / eps) / g1))
            nmin = minimal_nodes(c, 1.0, eps)
            rows.append(dict(eps=eps, c=c, hc_opt=nopt, hc_ap=nap, ours=nsuf, exact_min=nmin))
            print("nodes", eps, c, nopt, nap, nsuf, nmin, flush=True)
    write("quadrature_nodes.csv", rows)


def check_expansion():
    rows = []
    for c, alpha in [(0.2, 1.0), (0.5, 0.5), (1.0, 2.0)]:
        for nu in [0.0, 0.5, 1.0]:
            for m in [8, 16, 32, 64, 128, 256]:
                n = int(round(nu * m))
                B = lgf.reference(n, m, c, alpha, rtol=1e-15)
                A, b, beta, t1, t2 = lgf.expansion_data(n, m, c, alpha, 5)
                for J in range(1, 6):
                    approx = A * np.sum(b[:J] / float(m) ** np.arange(J))
                    rows.append(dict(c=c, alpha=alpha, nu=nu, n=n, m=m, J=J, rel_err=abs(approx - B) / B))
    write("expansion_error.csv", rows)
    slopes = {}
    for J in range(1, 5):
        sel = [r for r in rows if r["J"] == J and r["m"] >= 32]
        xs, ys = [], []
        for r in sel:
            xs.append(np.log(r["m"]))
            ys.append(np.log(r["rel_err"]))
        slopes[J] = float(np.polyfit(xs, ys, 1)[0])
    return slopes


def check_periodic():
    rows = []
    P = 32
    ck = lgf.periodic_modes(P)
    for (n1, n2) in [(4, 3), (8, 6), (16, 12), (24, 18), (32, 24), (48, 36), (64, 48)]:
        vals = [lgf.reference(n1, n2, ck[k], 1.0) for k in range(1, P)]
        for n3 in [0, 5]:
            R = sum(np.cos(2 * np.pi * k * n3 / P) * vals[k - 1] for k in range(1, P)) / P
            bound = (P - 1) / P * np.exp(-lgf.kappa(n1, n2, ck[1], 1.0)) * lgf.decay_constant(ck[1], 1.0)
            two = 2.0 / P * np.cos(2 * np.pi * n3 / P) * lgf.far_field(n1, n2, ck[1], 1.0, 4)
            rest = (P - 3) / P * np.exp(-lgf.kappa(n1, n2, ck[2], 1.0)) * lgf.decay_constant(ck[2], 1.0)
            rows.append(dict(n1=n1, n2=n2, n3=n3, remainder=R, bound=bound, two_mode=two,
                             two_mode_err=abs(R - two), second_bound=rest))
    write("periodic_far_field.csv", rows)
    return all(abs(r["remainder"]) <= r["bound"] for r in rows)


def check_bdt():
    rng = np.random.default_rng(7)
    rows = []
    for eps in [1e-6, 1e-9, 1e-12]:
        worst = 0.0
        for _ in range(40):
            c = float(10 ** rng.uniform(-2.5, 0.7))
            alpha = float(10 ** rng.uniform(-0.7, 0.7))
            L1 = int(rng.integers(4, 300))
            L2 = int(rng.integers(4, 300))
            R = lgf.reference_table(c, alpha, L1, L2)
            T = lgf.expand(lgf.bdt(c, alpha, L1, L2, eps), L1, L2)
            e = float(np.abs(T - R).max())
            worst = max(worst, e)
            rows.append(dict(eps=eps, c=c, alpha=alpha, L1=L1, L2=L2, err=e))
        print("bdt eps", eps, "worst", worst, flush=True)
    write("bdt_accuracy.csv", rows)
    return {e: max(r["err"] for r in rows if r["eps"] == e) for e in [1e-6, 1e-9, 1e-12]}


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    summary["decay_max_ratio"] = check_decay()
    print("decay", summary["decay_max_ratio"], flush=True)
    summary["aliasing_max_rel_dev"] = check_aliasing()
    print("aliasing", summary["aliasing_max_rel_dev"], flush=True)
    summary["expansion_slopes"] = check_expansion()
    print("slopes", summary["expansion_slopes"], flush=True)
    summary["periodic_bound_holds"] = bool(check_periodic())
    print("periodic", summary["periodic_bound_holds"], flush=True)
    summary["bdt_worst_error"] = {str(k): v for k, v in check_bdt().items()}
    check_nodes()
    with open(os.path.join(OUT, "verification_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
