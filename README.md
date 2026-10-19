# Efficient Tabulation of Screened Lattice Green's Functions

This repository contains our implementation or research on
*Sharp Decay, Far Field Expansion, and Optimal Tabulation of Screened Lattice Green's Functions*.
It computes the lattice Green's function (LGF) `B_c(n, m)` of the five point screened Poisson operator
```
$(L_c u)(n,m) = c^2 u(n,m) + alpha [2u(n,m) - u(n-1,m) - u(n+1,m)] + [2u(n,m) - u(n,m-1) - u(n,m+1)]$
```
on the lattice `Z^2`, with screening `c > 0` and anisotropy `alpha > 0`, and it reproduces results of our research.

## Files and Contents

| path | content |
| --- | --- |
| `lgf/core.py` | decay norm `kappa_c` and the constant `C_{c,alpha}` (Theorem 3.4), trapezoidal sums and node counts (Corollary 3.5), the pruned Cooley and Tukey evaluation of edge sums, the shifted contour reference with relative accuracy, the far field expansion coefficients (Theorem 4.1), and the exponentially convergent Bessel rule (Lemma 5.2) |
| `lgf/bdt.py` | the boundary Dirichlet tabulation `bdt(c, alpha, L1, L2, eps)` (Algorithm 1) and `expand` |
| `lgf/baselines.py` | the seven existing methods: Bessel quadrature (BQ), path series (PS), trapezoid and FFT (TFFT), adaptive trapezoid (ATR), 2D periodization (PFFT), hypergeometric series (HG), and the marching recurrence (REC) |
| `lgf/bench.py` | timing (best of repeated runs) and peak memory (`tracemalloc`) |
| `scripts/verify.py` | numerical checks of Theorem 3.4, Corollary 3.5, Theorem 4.1, Corollary 4.2 and Theorem 5.3 |
| `scripts/benchmark.py` | time, memory and accuracy of all eight methods |
| `scripts/figures.py` | EPS figures of the research |
| `results/` | CSV and JSON outputs, logs, and generated tables |


## Running the Experiments

Python 3.10 or newer is required, along with numpy>=2.0, scipy>=1.13, mpmath>=1.3, and matplotlib>=3.8.

To have a quick use: 
```python
import lgf

T = lgf.bdt(0.05, 1.0, 255, 255, 1e-12)
B = lgf.expand(T, 255, 255)
print(B[0, 0], B[10, 7])
print(lgf.decay_bound(10, 7, 0.05, 1.0))
print(lgf.far_field(200, 150, 0.05, 1.0, 4), lgf.reference(200, 150, 0.05, 1.0))
```
`bdt` returns the table on its `eps` support, a box of at most `(L1+1) x (L2+1)` entries; `expand` pads it with
zeros, which are within `eps` of the true values by Theorem 3.4.

```bash
./run_all.sh
```
runs, in order,

1. `python3 scripts/verify.py` (about 5 minutes), which writes `results/decay_ratio.csv`,
   `results/aliasing_identity.csv`, `results/expansion_error.csv`, `results/periodic_far_field.csv`,
   `results/bdt_accuracy.csv`, `results/quadrature_nodes.csv` and `results/verification_summary.json`;
2. `python3 scripts/benchmark.py` (about 60 minutes), which writes `results/benchmark_sizes.csv` (sides 64 to 2048,
   five pairs `(c, alpha)`), `results/benchmark_small.csv` (all eight methods on the side 32 with `c = 1`) and
   `results/benchmark_sweep.csv` (side 256, `c` from `1e-4` to `3`); the option `--quick` runs a reduced set and the
   option `--part sizes|small|sweep` runs one part;
3. `python3 scripts/tables.py`, which writes `results/table_nodes.tex` and `results/table_benchmark.tex`;
4. `python3 scripts/figures.py`, which writes `figures/decay_expansion.eps`, `figures/benchmark.eps`,
   `figures/sweep.eps` and `figures/surface3d.eps`; `epstopdf` converts them to the PDF files used by the paper.

All runs are single threaded (`run_all.sh` sets the thread variables of the BLAS libraries to one).

