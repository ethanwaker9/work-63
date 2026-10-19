import numpy as np
from math import factorial
from scipy import fft as sfft
from scipy.optimize import brentq


def gammas(c, alpha):
    return 2.0 * np.arcsinh(c / (2.0 * np.sqrt(alpha))), 2.0 * np.arcsinh(0.5 * c)


def decay_constant(c, alpha):
    return 1.0 / (np.sqrt(2.0) * c * min(1.0, np.sqrt(alpha)))


def support_radius(c, alpha, eps):
    return np.log(decay_constant(c, alpha) / eps)


def support_box(c, alpha, eps):
    g1, g2 = gammas(c, alpha)
    rho = support_radius(c, alpha, eps)
    if rho <= 0.0:
        return 1, 1
    return int(np.ceil(rho / g1)), int(np.ceil(rho / g2))


def boundary_point(phi, c, alpha):
    t1 = 2.0 * np.arcsinh(c / (2.0 * np.sqrt(alpha)) * np.sin(phi))
    t2 = 2.0 * np.arcsinh(0.5 * c * np.cos(phi))
    return t1, t2


def saddle(n, m, c, alpha):
    n = abs(float(n))
    m = abs(float(m))
    if n == 0.0:
        return boundary_point(0.0, c, alpha)
    if m == 0.0:
        return boundary_point(0.5 * np.pi, c, alpha)

    def g(phi):
        t1, t2 = boundary_point(phi, c, alpha)
        return n * np.sinh(t2) - m * alpha * np.sinh(t1)

    phi = brentq(g, 0.0, 0.5 * np.pi, xtol=1e-15, rtol=1e-15, maxiter=500)
    return boundary_point(phi, c, alpha)


def kappa(n, m, c, alpha):
    t1, t2 = saddle(n, m, c, alpha)
    return abs(n) * t1 + abs(m) * t2


def kappa_grid(n, m, c, alpha, iters=80):
    n = np.abs(np.asarray(n, dtype=float))
    m = np.abs(np.asarray(m, dtype=float))
    n, m = np.broadcast_arrays(n, m)
    lo = np.zeros(n.shape)
    hi = np.full(n.shape, 0.5 * np.pi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        t1, t2 = boundary_point(mid, c, alpha)
        pos = n * np.sinh(t2) - m * alpha * np.sinh(t1) > 0.0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    t1, t2 = boundary_point(0.5 * (lo + hi), c, alpha)
    return n * t1 + m * t2


def decay_bound(n, m, c, alpha):
    return np.exp(-kappa(n, m, c, alpha)) * decay_constant(c, alpha)


def half_shift(theta, c, alpha):
    w1 = 0.5 * c * c + 2.0 * alpha * np.sin(0.5 * theta) ** 2
    r = np.sqrt(w1) * np.sqrt(w1 + 2.0)
    return w1, r


def symbol_row(theta, m, c, alpha):
    w1, r = half_shift(theta, c, alpha)
    return np.exp(-abs(m) * np.log1p(w1 + r)) / (2.0 * r)


def log_cosh(x):
    x = abs(x)
    return x + np.log1p(np.exp(-2.0 * x)) - np.log(2.0)


def nodes_for_rows(nmax, gamma, C, eps):
    lhs = np.logaddexp(0.0, np.log(2.0 * C / eps) + log_cosh(gamma * nmax))
    N = int(np.ceil(lhs / gamma))
    N = max(N, 2 * (nmax + 1))
    N += N % 2
    return N


def trap_row(m, c, alpha, N):
    theta = 2.0 * np.pi * np.arange(N // 2 + 1) / N
    return sfft.irfft(symbol_row(theta, m, c, alpha), N)


def trap_row_pruned(m, c, alpha, N, nmax, budget):
    P = 2
    while P < 2 * (nmax + 1):
        P *= 2
    Q = -(-N // P)
    N = P * Q
    h = 2.0 * np.pi / N
    chunk = max(1, budget // (4 * P))
    k1 = (Q * h) * np.arange(P)
    n = np.arange(nmax + 1, dtype=float)
    acc = np.zeros(nmax + 1, dtype=complex)
    for a in range(0, Q, chunk):
        k2 = h * np.arange(a, min(a + chunk, Q))
        S = symbol_row(np.add.outer(k2, k1), m, c, alpha)
        X = sfft.rfft(S, axis=1)[:, :nmax + 1]
        acc += np.einsum("ij,ij->j", np.conjugate(X), np.exp(1j * np.multiply.outer(k2, n)))
    return acc.real / N


def reference_table(c, alpha, L1, L2, eps=1e-18):
    g1, _ = gammas(c, alpha)
    N = nodes_for_rows(L1, g1, decay_constant(c, alpha), eps)
    N = sfft.next_fast_len(N)
    N += N % 2
    theta = 2.0 * np.pi * np.arange(N // 2 + 1) / N
    w1, r = half_shift(theta, c, alpha)
    lk = np.log1p(w1 + r)
    base = 1.0 / (2.0 * r)
    T = np.empty((L1 + 1, L2 + 1))
    for m in range(L2 + 1):
        T[:, m] = sfft.irfft(np.exp(-m * lk) * base, N)[:L1 + 1]
    return T


def shifted_trap(n, m, c, alpha, tau, N):
    x = 2.0 * np.pi * np.arange(N) / N
    w1, r = half_shift(x + 1j * tau, c, alpha)
    F = np.exp(-abs(m) * np.log1p(w1 + r)) / (2.0 * r)
    return np.exp(-abs(n) * tau) * np.real(np.mean(np.exp(1j * abs(n) * x) * F))


def reference(n, m, c, alpha, rtol=1e-14, nmax_log2=24):
    n, m = abs(int(n)), abs(int(m))
    t1, t2 = saddle(n, m, c, alpha)
    if m >= n:
        f = lambda N: shifted_trap(n, m, c, alpha, t1, N)
    else:
        f = lambda N: shifted_trap(m, n, c / np.sqrt(alpha), 1.0 / alpha, t2, N) / alpha
    N = 64
    prev = f(N)
    while True:
        N *= 2
        cur = f(N)
        if abs(cur - prev) <= rtol * abs(cur) or N >= 2 ** nmax_log2:
            return cur
        prev = cur


def taylor(fun, deg, r0, M=512):
    phi = 2.0 * np.pi * np.arange(M) / M
    v = fun(r0 * np.exp(1j * phi))
    co = np.fft.fft(v) / M
    return np.array([co[k] / r0 ** k for k in range(deg + 1)])


def double_factorial(k):
    r = 1
    while k > 1:
        r *= k
        k -= 2
    return r


def expansion_data(n, m, c, alpha, J):
    n, m = abs(n), abs(m)
    nu = n / m
    t1, t2 = saddle(n, m, c, alpha)
    g1, _ = gammas(c, alpha)
    r0 = 0.5 * (g1 - t1)

    def s_of(x):
        w1, r = half_shift(x + 1j * t1, c, alpha)
        return np.log1p(w1 + r), 2.0 * r

    psi = lambda x: s_of(x)[0] - t2 - 1j * nu * x
    amp = lambda x: 1.0 / s_of(x)[1]
    D = max(6 * (J - 1), 2)
    p = taylor(psi, D, r0)
    q = taylor(amp, D, r0)
    beta = 2.0 * p[2].real
    chi = p.copy()
    chi[:3] = 0.0
    C = np.zeros(J, complex)
    chik = np.zeros(D + 1, complex)
    chik[0] = 1.0
    for k in range(0, 2 * (J - 1) + 1):
        prod = np.convolve(chik, q)[:D + 1] * ((-1) ** k / factorial(k))
        for j in range(J):
            l = 2 * (j + k)
            if l <= D and k <= 2 * j:
                C[j] += prod[l] * double_factorial(l - 1) / beta ** (j + k)
        chik = np.convolve(chik, chi)[:D + 1]
    b = (C / C[0]).real
    kap = n * t1 + m * t2
    A = np.exp(-kap) / (2.0 * np.sinh(t2) * np.sqrt(2.0 * np.pi * m * beta))
    return A, b, beta, t1, t2


def far_field(n, m, c, alpha, J):
    n, m = abs(int(n)), abs(int(m))
    if m >= n and m > 0:
        A, b, _, _, _ = expansion_data(n, m, c, alpha, J)
        return A * np.sum(b / float(m) ** np.arange(J))
    A, b, _, _, _ = expansion_data(m, n, c / np.sqrt(alpha), 1.0 / alpha, J)
    return A * np.sum(b / float(n) ** np.arange(J)) / alpha


def periodic_modes(P, a3=1.0):
    k = np.arange(P)
    return 2.0 * np.sqrt(a3) * np.sin(np.pi * k / P)


def bessel_nodes(c, alpha, eps):
    C = decay_constant(c, alpha)
    h = min(1.0, (2.0 * np.pi ** 2 / 3.0) / np.log1p(16.0 * C / eps))
    xlo = np.log(eps / 8.0)
    tp = max(1.0, np.log1p(np.pi / (np.sqrt(alpha) * eps))) / (c * c)
    xhi = h + np.log(tp)
    k0 = int(np.floor(xlo / h))
    k1 = int(np.ceil(xhi / h))
    return h, np.arange(k0, k1 + 1) * h


def scaled_bessel(nu, z):
    from scipy.special import ive
    z = float(z)
    if z <= 1e9:
        return ive(nu, z)
    nu = np.asarray(nu, dtype=float)
    mu = 4.0 * nu * nu
    term = np.ones_like(nu)
    total = np.ones_like(nu)
    for k in range(1, 9):
        term = term * (-(mu - (2.0 * k - 1.0) ** 2) / (8.0 * k * z))
        total = total + term
    return total / np.sqrt(2.0 * np.pi * z)


def edge_bessel(index, count, c, alpha, eps, along_first):
    h, x = bessel_nodes(c, alpha, eps)
    t = np.exp(x)
    wts = h * t * np.exp(-c * c * t)
    j = np.arange(count, dtype=float)
    out = np.zeros(count)
    for tq, wq in zip(t, wts):
        if wq == 0.0:
            continue
        if along_first:
            out += (wq * scaled_bessel(index, 2.0 * tq)) * scaled_bessel(j, 2.0 * alpha * tq)
        else:
            out += (wq * scaled_bessel(index, 2.0 * alpha * tq)) * scaled_bessel(j, 2.0 * tq)
    return out
