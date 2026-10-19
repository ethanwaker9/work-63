import numpy as np
import mpmath as mp
from scipy import integrate, special
from scipy import fft as sfft
from scipy.special import gammaln
from .core import gammas, decay_constant, nodes_for_rows, trap_row, symbol_row


def bessel_quadrature(c, alpha, L1, L2, eps):
    T = np.zeros((L1 + 1, L2 + 1))
    c2 = c * c
    a2 = 2.0 * alpha
    for n in range(L1 + 1):
        for m in range(L2 + 1):
            f = lambda t: np.exp(-c2 * t) * special.ive(n, a2 * t) * special.ive(m, 2.0 * t)
            T[n, m] = integrate.quad(f, 0.0, np.inf, epsabs=eps, epsrel=0.0, limit=200)[0]
    return T


def series_terms(c, alpha, eps):
    lam = 2.0 + 2.0 * alpha
    q = lam / (lam + c * c)
    return int(np.ceil(np.log(c * c * eps) / np.log(q)))


def path_series(c, alpha, L1, L2, eps):
    lam = 2.0 + 2.0 * alpha
    K = series_terms(c, alpha, eps)
    T = np.zeros((L1 + 1, L2 + 1))
    la = np.log(alpha)
    lA = np.log(lam + c * c)
    lg = gammaln(np.arange(K + 2) + 1.0)
    for n in range(L1 + 1):
        for m in range(L2 + 1):
            tot = 0.0
            for k in range(n + m, K, 2):
                h = (k - n - m) // 2
                l = np.arange(h + 1)
                lt = (n + 2 * l) * la + lg[k] - lg[l] - lg[n + l] - lg[h - l] - lg[h - l + m] - (k + 1) * lA
                tot += np.exp(lt).sum()
            T[n, m] = tot
    return T


def trapezoid_fft(c, alpha, L1, L2, eps, delta=0.01):
    eta = (1.0 - delta) ** 2 * c * c / alpha
    geta = np.log(1.0 + 0.5 * eta + np.sqrt((1.0 + 0.5 * eta) ** 2 - 1.0))
    Np = int(np.ceil(np.log(1.0 / (eps * (c / np.sqrt(alpha)) * np.sqrt(2.0 * delta - delta ** 2))) / (2.0 * geta))) + L1
    Np = max(Np, L1 + 1)
    theta = np.pi * np.arange(Np) / Np
    sgn = (-1.0) ** np.arange(L1 + 1)
    T = np.empty((L1 + 1, L2 + 1))
    for m in range(L2 + 1):
        v = symbol_row(theta, m, c, alpha)
        fpi = symbol_row(np.pi, m, c, alpha)
        T[:, m] = sfft.irfft(v, 2 * Np)[:L1 + 1] + sgn * fpi / (2 * Np)
    return T


def adaptive_trapezoid(c, alpha, L1, L2, eps, nmax_log2=24):
    T = np.zeros((L1 + 1, L2 + 1))
    for n in range(L1 + 1):
        N0 = 1 << int(np.ceil(np.log2(4 * (n + 1))))
        for m in range(L2 + 1):
            N = N0
            f = lambda th: np.cos(n * th) * symbol_row(th, m, c, alpha)
            S = f(2.0 * np.pi * np.arange(N) / N).sum()
            I = S / N
            while True:
                S += f(2.0 * np.pi * (np.arange(N) + 0.5) / N).sum()
                N *= 2
                In = S / N
                if abs(In - I) <= eps or N >= 2 ** nmax_log2:
                    break
                I = In
            T[n, m] = In
    return T


def periodized_fft2(c, alpha, L1, L2, eps):
    g1, g2 = gammas(c, alpha)
    lc = np.log(16.0 * decay_constant(c, alpha) / eps)
    P1 = sfft.next_fast_len(L1 + 1 + int(np.ceil(lc / g1)), real=True)
    P2 = sfft.next_fast_len(L2 + 1 + int(np.ceil(lc / g2)), real=True)
    P1 += P1 % 2
    P2 += P2 % 2
    k1 = 2.0 * np.pi * np.arange(P1) / P1
    k2 = 2.0 * np.pi * np.arange(P2 // 2 + 1) / P2
    S = c * c + 4.0 * alpha * np.sin(0.5 * k1)[:, None] ** 2 + 4.0 * np.sin(0.5 * k2)[None, :] ** 2
    np.reciprocal(S, out=S)
    G = sfft.irfft2(S, s=(P1, P2), overwrite_x=True)
    return G[:L1 + 1, :L2 + 1].copy()


def hypergeometric(c, alpha, L1, L2, eps, dps=20):
    mp.mp.dps = dps
    a = mp.mpf(1) + mp.mpf(alpha) + mp.mpf(c) ** 2 / 2
    x = (mp.mpf(alpha) / a) ** 2
    y = (1 / a) ** 2
    T = np.zeros((L1 + 1, L2 + 1))
    for n in range(L1 + 1):
        for m in range(L2 + 1):
            pre = mp.factorial(n + m) * mp.mpf(alpha) ** n / (mp.mpf(2) ** (n + m + 1) * a ** (n + m + 1) * mp.factorial(n) * mp.factorial(m))
            F = mp.appellf4(mp.mpf(n + m + 1) / 2, mp.mpf(n + m) / 2 + 1, n + 1, m + 1, x, y)
            T[n, m] = float(pre * F)
    return T


def marching(c, alpha, L1, L2, eps):
    g1, _ = gammas(c, alpha)
    W = L1 + L2 + 2
    N = sfft.next_fast_len(nodes_for_rows(W, g1, decay_constant(c, alpha), eps), real=True)
    N += N % 2
    r0 = trap_row(0, c, alpha, N)[:W + 1]
    r1 = trap_row(1, c, alpha, N)[:W + 1]
    T = np.zeros((L1 + 1, L2 + 1))
    T[:, 0] = r0[:L1 + 1]
    if L2 >= 1:
        T[:, 1] = r1[:L1 + 1]
    prev = np.concatenate([r0[:0:-1], r0])
    cur = np.concatenate([r1[:0:-1], r1])
    d = c * c + 2.0 * alpha + 2.0
    for m in range(1, L2):
        nxt = d * cur[1:-1] - alpha * (cur[:-2] + cur[2:]) - prev[1:-1]
        prev = cur[1:-1]
        cur = nxt
        z = (cur.size - 1) // 2
        T[:, m + 1] = cur[z:z + L1 + 1]
    return T
