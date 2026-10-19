import numpy as np
from scipy import fft as sfft
from scipy.linalg import blas
from .core import gammas, decay_constant, nodes_for_rows, trap_row, trap_row_pruned, bessel_nodes, edge_bessel


def edge_values(index, count, c, alpha, eps, budget, first):
    if first:
        cc, aa, ee, sc = c, alpha, eps, 1.0
    else:
        cc, aa, ee, sc = c / np.sqrt(alpha), 1.0 / alpha, alpha * eps, 1.0 / alpha
    g, _ = gammas(cc, aa)
    N = nodes_for_rows(count - 1, g, decay_constant(cc, aa), ee)
    Q = bessel_nodes(c, alpha, eps)[1].size
    if 4 * Q * count < N:
        return edge_bessel(index, count, c, alpha, eps, first)
    if 2 * N <= budget:
        out = trap_row(index, cc, aa, N)[:count].copy()
    else:
        out = trap_row_pruned(index, cc, aa, N, count - 1, budget)
    out *= sc
    return out


def bdt(c, alpha, L1, L2, eps):
    g1, g2 = gammas(c, alpha)
    rho = np.log(decay_constant(c, alpha) / eps)
    if rho <= 0.0:
        return np.zeros((1, 1))
    K1 = int(np.ceil(rho / g1))
    K2 = int(np.ceil(rho / g2))
    N1 = min(L1 + 1, K1)
    N2 = min(L2 + 1, K2)
    budget = max((N1 * N2) // 2, 1024)
    G1 = None
    G2 = None
    if N1 < K1:
        G1 = sfft.dct(edge_values(N1, N2, c, alpha, eps, budget, False), type=3, overwrite_x=True)
    if N2 < K2:
        G2 = sfft.dct(edge_values(N2, N1, c, alpha, eps, budget, True), type=3, overwrite_x=True)
    w1 = np.arange(N1, dtype=float)
    w1 *= np.pi / N1
    w1 += 0.5 * np.pi / N1
    d1 = np.sin(0.5 * w1)
    d1 *= d1
    d1 *= 4.0 * alpha
    d1 += c * c
    scale = 0.25 / (N1 * N2)
    h2 = np.pi / N2
    if G1 is None and G2 is None:
        d2 = np.arange(N2, dtype=float)
        d2 *= h2
        d2 += 0.5 * h2
        np.sin(0.5 * d2, out=d2)
        d2 *= d2
        d2 *= 4.0
        U = np.zeros((N1, N2), order="F")
        w1.fill(1.0)
        U = blas.dger(1.0, d1, np.ones(N2), a=U, overwrite_a=1)
        U = blas.dger(1.0, w1, d2, a=U, overwrite_a=1)
        np.reciprocal(U, out=U)
        U *= scale
    else:
        w1 *= N1 - 1
        np.cos(w1, out=w1)
        w1 *= 2.0 * alpha * scale
        if G1 is None:
            G1 = np.zeros(N2)
        if G2 is None:
            G2 = np.zeros(N1)
        G2 *= scale
        U = np.empty((N1, N2), order="F")
        num = np.empty(N1)
        for k in range(N2):
            wk = (k + 0.5) * h2
            sk = np.sin(0.5 * wk)
            col = U[:, k]
            np.add(d1, 4.0 * sk * sk, out=col)
            num.fill(scale)
            num = blas.daxpy(w1, num, a=G1[k])
            num = blas.daxpy(G2, num, a=2.0 * np.cos(wk * (N2 - 1)))
            np.divide(num, col, out=col)
    return sfft.dctn(U, type=2, overwrite_x=True)


def expand(T, L1, L2):
    out = np.zeros((L1 + 1, L2 + 1))
    a = min(T.shape[0], L1 + 1)
    b = min(T.shape[1], L2 + 1)
    out[:a, :b] = T[:a, :b]
    return out
