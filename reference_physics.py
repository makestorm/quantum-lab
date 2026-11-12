# -*- coding: utf-8 -*-
"""Independent reference implementation for QuantumLab (red-team verifier).

This module deliberately does NOT import ``qllab.models`` (or any helper that
builds a Hamiltonian for the product code).  All matrices and all time
evolutions here are built from scratch with numpy, using algorithms that are
mathematically independent of the spectral decomposition
``U(t) = V exp(-i Lambda t) V^dagger`` used by the implementation under test:

  (a) ``expm`` -- Taylor series of ``exp(A)`` with scaling & squaring
      (>= 40 Taylor terms after scaling; independent of eigendecomposition),
  (b) ``rk4_evolve`` -- fixed-step classical RK4 integration of
      ``i dpsi/dt = H psi``,
  (c) closed-form analytic solutions (Bloch precession, Fock-space
      ``<x>(t) = sqrt(2) Re[A0 exp(-i omega t)]``, ``E0 = -sqrt(D^2+L^2)/2``).

Conventions (must match CONTRACT.md / references/models.md):
  * hbar = 1, basis |up>=(1,0), |down>=(0,1),
  * spin:  H = (1/2)(Ox sx + Oy sy + Oz sz),
  * HO:    H = omega (a^dag a + 1/2) truncated to N Fock levels,
           a[n-1, n] = sqrt(n),
  * pert:  H = (D/2) sz + (L/2) sx,  ground state |down> (sz = -1).

Run ``python tests/reference_physics.py`` for a self-consistency check
(Taylor vs RK4 vs closed form) that does not need the product code.
"""
from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------
# Pauli matrices, built explicitly (no dependency on the product code)
# --------------------------------------------------------------------------
SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# --------------------------------------------------------------------------
# (a) exp(A) by Taylor series with scaling & squaring
# --------------------------------------------------------------------------
def _taylor_small(A: np.ndarray, terms: int) -> np.ndarray:
    """sum_{k=0..terms} A^k / k!  (Horner-style, no eigendecomposition)."""
    n = A.shape[0]
    S = np.eye(n, dtype=complex)
    term = np.eye(n, dtype=complex)
    for k in range(1, terms + 1):
        term = term @ A / float(k)
        S = S + term
    return S


def expm(A: np.ndarray, terms: int = 48) -> np.ndarray:
    """Matrix exponential, Taylor + scaling/squaring.

    After scaling ||A/2^s||_inf <= 0.5, a 48-term Taylor series is accurate to
    ~1e-60; repeated squaring keeps the result at machine precision.
    """
    A = np.asarray(A, dtype=complex)
    nrm = np.linalg.norm(A, ord=np.inf)
    s = 0
    while nrm > 0.5:
        nrm /= 2.0
        s += 1
    E = _taylor_small(A / float(2 ** s), terms)
    for _ in range(s):
        E = E @ E
    return E


def evolve_expm(H: np.ndarray, psi0: np.ndarray, times) -> np.ndarray:
    """psi(t) = exp(-i H (t - t0)) psi0 for every t in ``times``."""
    H = np.asarray(H, dtype=complex)
    psi0 = np.asarray(psi0, dtype=complex)
    t0 = float(times[0])
    return np.array([expm(-1j * H * (float(t) - t0)) @ psi0 for t in times],
                    dtype=complex)


# --------------------------------------------------------------------------
# (b) fixed-step RK4 for i dpsi/dt = H psi
# --------------------------------------------------------------------------
def rk4_evolve(H: np.ndarray, psi0: np.ndarray, times, substeps: int = 1) -> np.ndarray:
    """Fixed-step classical RK4 on i dpsi/dt = H psi.

    ``times`` must be (close to) uniform; ``substeps`` RK4 steps are taken per
    output interval.  Returns the state at every requested time.
    """
    H = np.asarray(H, dtype=complex)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be 1-D with >= 2 points")
    dts = np.diff(times)
    if np.max(np.abs(dts - dts[0])) > 1e-12 * max(1.0, abs(dts[0])):
        raise ValueError("rk4_evolve requires a uniform time grid")
    h = float(dts[0]) / float(substeps)

    def f(psi):
        return -1j * (H @ psi)

    psi = np.asarray(psi0, dtype=complex).copy()
    out = [psi.copy()]
    for _ in range(times.size - 1):
        for _ in range(substeps):
            k1 = f(psi)
            k2 = f(psi + 0.5 * h * k1)
            k3 = f(psi + 0.5 * h * k2)
            k4 = f(psi + h * k3)
            psi = psi + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out.append(psi.copy())
    return np.array(out, dtype=complex)


# --------------------------------------------------------------------------
# Hamiltonians (independent construction)
# --------------------------------------------------------------------------
def spin_hamiltonian(ox: float, oy: float, oz: float) -> np.ndarray:
    return 0.5 * (float(ox) * SX + float(oy) * SY + float(oz) * SZ)


def omega_vec(ox: float, oy: float, oz: float) -> np.ndarray:
    return np.array([float(ox), float(oy), float(oz)])


def spin_initial_state(kind: str, vector=None) -> np.ndarray:
    if kind == "up":
        return np.array([1.0, 0.0], dtype=complex)
    if kind == "down":
        return np.array([0.0, 1.0], dtype=complex)
    if kind == "plus_x":
        return np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
    if kind == "plus_y":
        return np.array([1.0, 1.0j], dtype=complex) / math.sqrt(2.0)
    if kind == "custom":
        if vector is None:
            raise ValueError("custom state needs a vector")
        v = np.array([complex(*p) if len(p) == 2 else complex(p[0])
                      for p in vector], dtype=complex)
        return v / np.linalg.norm(v)
    raise ValueError(f"unknown spin initial state {kind!r}")


# --------------------------------------------------------------------------
# (c) closed-form spin analytics
# --------------------------------------------------------------------------
def spin_bloch(ox: float, oy: float, oz: float, t, n0) -> np.ndarray:
    """Closed-form Bloch precession n(t); returns array (len(t), 3).

    n(t) = n_par * Om_hat
           + [n0 - n_par Om_hat] cos(w t)
           + (Om_hat x n0) sin(w t),   w = |Om|,  n_par = Om_hat . n0
    """
    Om = omega_vec(ox, oy, oz)
    w = float(np.linalg.norm(Om))
    if w == 0.0:
        raise ValueError("|Omega| = 0 has no dynamics")
    u = Om / w
    n0 = np.asarray(n0, dtype=float)
    n_par = float(np.dot(u, n0))
    perp = n0 - n_par * u
    cross = np.cross(u, n0)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    c = np.cos(w * t)[:, None]
    s = np.sin(w * t)[:, None]
    return n_par * u[None, :] + perp[None, :] * c + cross[None, :] * s


def spin_pops_from_bloch(ox: float, oy: float, oz: float, t, n0):
    """P_up = (1+nz)/2, P_down = (1-nz)/2 from the closed form."""
    n = spin_bloch(ox, oy, oz, t, n0)
    return (1.0 + n[:, 2]) / 2.0, (1.0 - n[:, 2]) / 2.0


def spin_flip_prob_a1(ox: float, oy: float, oz: float, t):
    """P_down(t) for initial |up>: (Ox^2+Oy^2)/|Om|^2 * sin^2(|Om| t / 2)."""
    Om = omega_vec(ox, oy, oz)
    w = float(np.linalg.norm(Om))
    wperp2 = float(ox) ** 2 + float(oy) ** 2
    t = np.asarray(t, dtype=float)
    return (wperp2 / w ** 2) * np.sin(w * t / 2.0) ** 2


def spin_flip_peak(ox: float, oy: float, oz: float) -> float:
    """Theoretical maximum of P_down for initial |up> (analytic)."""
    wperp2 = float(ox) ** 2 + float(oy) ** 2
    w2 = wperp2 + float(oz) ** 2
    return wperp2 / w2


def spin_period(ox: float, oy: float, oz: float) -> float:
    return 2.0 * math.pi / math.sqrt(float(ox) ** 2 + float(oy) ** 2 + float(oz) ** 2)


# --------------------------------------------------------------------------
# Harmonic oscillator (independent construction)
# --------------------------------------------------------------------------
def ho_ladder(n_levels: int):
    """Return (a, a_dag) with a[n-1, n] = sqrt(n), n = 1..N-1."""
    N = int(n_levels)
    a = np.zeros((N, N), dtype=complex)
    for n in range(1, N):
        a[n - 1, n] = math.sqrt(n)
    return a, a.conj().T


def ho_hamiltonian(omega: float, n_levels: int) -> np.ndarray:
    a, ad = ho_ladder(n_levels)
    return float(omega) * (ad @ a + 0.5 * np.eye(int(n_levels), dtype=complex))


def ho_position(n_levels: int) -> np.ndarray:
    a, ad = ho_ladder(n_levels)
    return (a + ad) / math.sqrt(2.0)


def ho_fock_state(n_levels: int, n: int) -> np.ndarray:
    v = np.zeros(int(n_levels), dtype=complex)
    v[int(n)] = 1.0
    return v


def ho_superposition(n_levels: int, components) -> np.ndarray:
    """components: [(n, (re, im)), ...]; normalised here."""
    v = np.zeros(int(n_levels), dtype=complex)
    for n, amp in components:
        v[int(n)] += complex(*amp) if len(amp) == 2 else complex(amp[0])
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        raise ValueError("zero superposition vector")
    return v / nrm


def ho_coherent(n_levels: int, alpha):
    """Truncated coherent state c_n = e^{-|a|^2/2} a^n/sqrt(n!), renormalised.

    Returns (psi, truncation_leak) with leak = 1 - sum |c_n|^2 *before*
    renormalisation (the physically meaningful truncation error).
    """
    N = int(n_levels)
    al = complex(*alpha) if len(alpha) == 2 else complex(alpha[0])
    c = np.zeros(N, dtype=complex)
    log_pref = -abs(al) ** 2 / 2.0
    for n in range(N):
        c[n] = math.exp(log_pref) * (al ** n) / math.sqrt(math.factorial(n))
    s = float(np.sum(np.abs(c) ** 2))
    leak = 1.0 - s
    if s <= 0.0:
        raise ValueError("coherent state completely outside truncation")
    return c / math.sqrt(s), leak


def ho_analytic_expect_x(omega: float, t, psi0) -> np.ndarray:
    """<x>(t) = sqrt(2) Re[ A0 exp(-i omega t) ],  A0 = sum c_n* c_{n+1} sqrt(n+1)."""
    c = np.asarray(psi0, dtype=complex)
    N = c.size
    A0 = complex(0.0)
    for n in range(N - 1):
        A0 += np.conj(c[n]) * c[n + 1] * math.sqrt(n + 1)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.sqrt(2.0) * np.real(A0 * np.exp(-1j * float(omega) * t))


def ho_analytic_expect_p(omega: float, t, psi0) -> np.ndarray:
    """<p>(t) = sqrt(2) Im[ A0 exp(-i omega t) ]."""
    c = np.asarray(psi0, dtype=complex)
    N = c.size
    A0 = complex(0.0)
    for n in range(N - 1):
        A0 += np.conj(c[n]) * c[n + 1] * math.sqrt(n + 1)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.sqrt(2.0) * np.imag(A0 * np.exp(-1j * float(omega) * t))


def ho_expect_x(n_levels: int, psi) -> float:
    """<psi|x|psi> from the independently built x operator."""
    x = ho_position(n_levels)
    psi = np.asarray(psi, dtype=complex)
    return float(np.real(np.conj(psi) @ (x @ psi)))


# --------------------------------------------------------------------------
# Non-degenerate perturbation theory (closed form, no diagonalisation)
# --------------------------------------------------------------------------
def pert_hamiltonian(delta: float, lam: float) -> np.ndarray:
    """H = (D/2) sz + (L/2) sx."""
    return 0.5 * float(delta) * SZ + 0.5 * float(lam) * SX


def pert_energies(delta: float, lam: float) -> dict:
    D, L = float(delta), float(lam)
    exact = -0.5 * math.sqrt(D * D + L * L)
    pt1 = -0.5 * D                      # <0|V|0> = 0 (V has no diagonal)
    pt2 = -0.5 * D - L * L / (4.0 * D)  # -D/2 - L^2/(4D)
    abs_err = abs(pt2 - exact)
    return {
        "E0_exact": exact,
        "E0_pt1": pt1,
        "E0_pt2": pt2,
        "E1_exact": 0.5 * math.sqrt(D * D + L * L),
        "abs_err": abs_err,
        "rel_err": abs_err / abs(exact),
    }


def pert_states(delta: float, lam: float):
    """Closed-form eigenvectors of (D/2)sz + (L/2)sx."""
    D, L = float(delta), float(lam)
    r = math.sqrt(D * D + L * L)
    # H = [[D/2, L/2], [L/2, -D/2]]: eigenvalues -+r/2, eigenvectors
    # [L, -(D -+ r)] (row 1: D*L/2 - L(D -+ r)/2 = -+L*r/2).
    v_low = np.array([L, -(D + r)], dtype=complex)
    v_low /= np.linalg.norm(v_low)
    v_high = np.array([L, -(D - r)], dtype=complex)
    v_high /= np.linalg.norm(v_high)
    return v_low, v_high


# --------------------------------------------------------------------------
# Self-consistency check (no product code needed)
# --------------------------------------------------------------------------
def _maxdiff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def selftest(verbose: bool = True) -> dict:
    """Taylor vs RK4 vs closed form; returns dict of max deviations."""
    res = {}

    # --- spin: Taylor and RK4 against the Bloch closed form
    ox, oy, oz = 2.0, 0.0, 1.0
    H = spin_hamiltonian(ox, oy, oz)
    psi0 = spin_initial_state("up")
    T = spin_period(ox, oy, oz)
    t = np.linspace(0.0, T, 401)
    e = res

    psi_t = evolve_expm(H, psi0, t)
    psi_r = rk4_evolve(H, psi0, t, substeps=1)
    n_t = np.array([np.real(np.conj(p) @ SZ @ p) for p in psi_t])
    n_r = np.array([np.real(np.conj(p) @ SZ @ p) for p in psi_r])
    pd_theory = spin_flip_prob_a1(ox, oy, oz, t)
    pd_t = np.abs(psi_t[:, 1]) ** 2
    pd_r = np.abs(psi_r[:, 1]) ** 2
    e["spin_taylor_vs_analytic_pop"] = _maxdiff(pd_t, pd_theory)
    e["spin_taylor_vs_analytic_nz"] = _maxdiff(n_t, spin_bloch(ox, oy, oz, t, [0, 0, 1])[:, 2])
    e["spin_rk4_vs_analytic_pop"] = _maxdiff(pd_r, pd_theory)
    e["spin_rk4_vs_analytic_nz"] = _maxdiff(n_r, spin_bloch(ox, oy, oz, t, [0, 0, 1])[:, 2])
    e["spin_taylor_vs_rk4"] = _maxdiff(psi_t, psi_r)
    # norm and unitarity / time reversal
    e["spin_norm_dev"] = float(np.max(np.abs(np.sum(np.abs(psi_t) ** 2, axis=1) - 1.0)))
    U_p = expm(-1j * H * 0.37)
    U_m = expm(1j * H * 0.37)
    e["spin_Ut_Uminus_t_I"] = _maxdiff(U_p @ U_m, np.eye(2))
    # arbitrary-direction consistency (Oy != 0)
    H2 = spin_hamiltonian(0.7, -1.3, 0.4)
    psi02 = spin_initial_state("plus_x")
    n0 = np.array([1.0, 0.0, 0.0])
    ps2 = evolve_expm(H2, psi02, t)
    nz2 = np.array([np.real(np.conj(p) @ SZ @ p) for p in ps2])
    e["spin_oY_taylor_vs_bloch"] = _maxdiff(
        nz2, spin_bloch(0.7, -1.3, 0.4, t, n0)[:, 2])

    # --- HO: Taylor vs RK4 vs Fock-space closed form
    omega, N = 1.0, 40
    Hh = ho_hamiltonian(omega, N)
    xop = ho_position(N)
    psi_h = ho_superposition(N, [(0, (1.0, 0.0)), (1, (1.0, 0.0))])
    t_h = np.linspace(0.0, 2.0 * math.pi, 201)
    eh_t = evolve_expm(Hh, psi_h, t_h)
    eh_r = rk4_evolve(Hh, psi_h, t_h, substeps=8)
    xt = np.array([float(np.real(np.conj(p) @ (xop @ p))) for p in eh_t])
    xr = np.array([float(np.real(np.conj(p) @ (xop @ p))) for p in eh_r])
    xa = ho_analytic_expect_x(omega, t_h, psi_h)
    e["ho_taylor_vs_analytic_x"] = _maxdiff(xt, xa)
    e["ho_rk4_vs_analytic_x"] = _maxdiff(xr, xa)
    e["ho_taylor_vs_rk4"] = _maxdiff(eh_t, eh_r)
    e["ho_norm_dev"] = float(np.max(np.abs(np.sum(np.abs(eh_t) ** 2, axis=1) - 1.0)))
    ev = np.linalg.eigvalsh(Hh)
    e["ho_spectrum_vs_n_plus_half"] = _maxdiff(ev, np.arange(N) + 0.5)
    # coherent state |alpha=1>
    psi_c, leak = ho_coherent(N, (1.0, 0.0))
    eh_c = evolve_expm(Hh, psi_c, t_h)
    xc = np.array([float(np.real(np.conj(p) @ (xop @ p))) for p in eh_c])
    e["ho_coherent_leak"] = float(leak)
    e["ho_coherent_analytic_x"] = _maxdiff(xc, np.sqrt(2.0) * np.cos(t_h))
    e["ho_coherent_analytic_formula"] = _maxdiff(
        ho_analytic_expect_x(omega, t_h, psi_c), np.sqrt(2.0) * np.cos(t_h))

    # --- perturbation closed form vs independent eigendecomposition
    d, lam = 1.0, 0.1
    pe = pert_energies(d, lam)
    evals = np.sort(np.linalg.eigvalsh(pert_hamiltonian(d, lam)))
    e["pert_exact_closed_form_vs_eigh"] = abs(pe["E0_exact"] - evals[0])
    v_low, _ = pert_states(d, lam)
    e["pert_eigenvector_residual"] = _maxdiff(
        pert_hamiltonian(d, lam) @ v_low, pe["E0_exact"] * v_low)

    if verbose:
        for k in sorted(e):
            print(f"  {k:38s} {e[k]:.3e}")
    return e


if __name__ == "__main__":
    print("reference_physics self-consistency (Taylor / RK4 / closed form):")
    out = selftest()
    worst = max(out.values())
    print(f"\nworst deviation = {worst:.3e}")
    raise SystemExit(0 if worst < 1e-6 else 1)
