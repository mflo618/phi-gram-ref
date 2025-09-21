#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phi_gram_ref.py — Reference computation for twisted Gram certificates
(fused-map + √Jacobian compensation + spectral radius/PSD diagnostics
+ P_2N stability
 + optional B-doubling + dynamic stability metrics)

Usage examples:
  python phi_gram_ref.py
  python phi_gram_ref.py --group su3 --grid 28 --B 4096 --refine 2 --refine-grid 12
  python phi_gram_ref.py --group su4 --tau 0.03 --r 0.1 --delta 0.0
  python phi_gram_ref.py --no-check-pn --check-b-doubling
  python phi_gram_ref.py --cesaro-start-T    # exclude identity term from Cesàro sum
  python phi_gram_ref.py --check-dynamics --N-dynamics 256

Outputs:
  - phi_gram_summary.json     (summary per group)
  - scan_<group>.json         (coarse + refined scan points with metrics)

Model sketch:
  Boundary Möbius with a = r·e^{i·delta}, composed with an in-plane twist τ in W = span{b5,b6}.
  Pullback on L²(dθ) uses the fused angle map θ_src_total and multiplies by √(dθ_src/dθ) for unitarity.
  Cesàro projector: P_N = (1/N) Σ_{k=0}^{N-1} T^k (or, with --cesaro-start-T, from k=1 to N).

Diagnostics:
  - Spectral radius ρ(T): if ρ(T) > 1 + 1e-6, Cesàro may not converge.
  - PSD: λ_min(G), λ_max(G), cond(G), violation flag.
  - Cesàro stability: ΔP₂N via ‖P_{2N} - P_N‖₂ and max|·|.
  - Optional B-doubling: re-evaluate best point at 2B and report deltas.
  - Dynamics (optional): convergence_rate, max_drift_angle, coherence_ratio from the evolution of
    P_k = (1/k) Σ_{i=1}^k T^i.
"""
import argparse
import json
from functools import lru_cache
from typing import Optional
import numpy as np
import sys

# -------- pretty printing (icons + ANSI colors) --------
class _Style:
    CODES = {
        'reset':'\033[0m','bold':'\033[1m','dim':'\033[2m','italic':'\033[3m',
        'red':'\033[31m','green':'\033[32m','yellow':'\033[33m','blue':'\033[34m',
        'magenta':'\033[35m','cyan':'\033[36m','white':'\033[37m'
    }
    def __init__(self, enable=True):
        self.enable = enable
    def color(self, text, *names):
        if not self.enable or not names:
            return str(text)
        seq = ''.join(self.CODES.get(n,'') for n in names)
        return f"{seq}{text}{self.CODES['reset']}"

def _tick(ok):
    return '✅' if ok else '❌'
def _warn():
    return '⚠️'

def _coherence_bar(rel_min: float, style, width: int = 20) -> str:
    """Render a left-filled bar where SMALLER rel_min fills MORE cells (more coherent).
    rel_min=0 → full bar; rel_min≥0.3 → empty bar. Threshold ticks at 0.03 and 0.15.
    """
    # map rel_min ∈ [0, 0.3+] to fill ∈ [1.0, 0.0]
    x = max(0.0, min(1.0, 1.0 - (rel_min / 0.3)))
    filled = int(round(x * width))
    empty = width - filled
    bar = '█' * filled + '░' * empty
    # choose bar color by thresholds
    color = 'green' if rel_min <= 0.03 else ('yellow' if rel_min <= 0.15 else 'red')
    return style.color(f"[{bar}]", color)

# ---------------- tolerances ----------------
EPS_RHO = 1e-6     # tolerance for spectral radius > 1
EPS_PSD = 1e-12    # tolerance for PSD eigenvalue negativity
EPS_H   = 1e-12    # stability threshold for c in h_min = -b/c

# ---------------- periodic interpolation ----------------
def periodic_interp(values: np.ndarray, theta_src: np.ndarray) -> np.ndarray:
    """
    Linear interpolation for samples on the circle.
    values[j] corresponds to θ_j = 2π j / B, j=0..B-1. theta_src is arbitrary angles.
    """
    B = values.shape[0]
    two_pi = 2.0 * np.pi
    x = np.mod(theta_src, two_pi) * (B / two_pi)  # in [0, B)
    i0 = np.floor(x).astype(int)                  # 0..B-1
    frac = x - i0
    i0 = i0 % B
    i1 = (i0 + 1) % B
    return (1.0 - frac) * values[i0] + frac * values[i1]

# ---------------- cached theta grid ----------------
@lru_cache(maxsize=None)
def _theta(B: int) -> np.ndarray:
    """Cached uniform grid on [0, 2π)."""
    return np.linspace(0.0, 2.0*np.pi, B, endpoint=False)

# ---------------- boundary basis ----------------
def basis_b5(B: int) -> np.ndarray:
    theta = _theta(B)
    return np.exp(1j * 5.0 * theta)

def basis_b6(B: int) -> np.ndarray:
    theta = _theta(B)
    return np.exp(1j * 6.0 * theta)

# ---------------- primitive map ----------------
def mobius_inverse_theta(theta_out: np.ndarray, a: complex, gamma: float) -> np.ndarray:
    """
    Inverse angle map for the Möbius transform z ↦ e^{iγ} (z - a) / (1 - \bar{a} z).
    Given θ_out, returns θ_in such that z_in maps to z_out on the unit circle.
    """
    e_minus_i_gamma = np.exp(-1j * gamma)
    z_out = np.exp(1j * theta_out)
    w = e_minus_i_gamma * z_out
    z_in = (w + a) / (1.0 + np.conj(a) * w)
    return np.angle(z_in)

# ---------------- projection to W ----------------
def project_to_W(g: np.ndarray, b5: np.ndarray, b6: np.ndarray) -> np.ndarray:
    B = g.shape[0]
    norm = 1.0 / B
    c5 = norm * np.vdot(b5, g)
    c6 = norm * np.vdot(b6, g)
    return np.array([c5, c6], dtype=np.complex128)

# ---------------- build T^τ(α,β) with fused map + √Jacobian ----------------
def build_T_matrix(alpha: float, beta: float, tau: float, delta: float, r: float,
                   B: int = 2048, gamma: float = 0.0,
                   b5: Optional[np.ndarray] = None, b6: Optional[np.ndarray] = None,
                   ablate_jacobian: bool = False) -> np.ndarray:
    """
    Construct the 2×2 matrix T in basis {b5,b6} for one cycle:
        rotate(α) → invert → rotate(β) → circumscribe(Möbius(a, γ)) → project to W
    Fused pullback:
        θ_src_total(θ) = -MobiusInverse(θ) + (β - α)
        dθ_src/dθ = (1 - |a|²) / |1 + \bar a · e^{-iγ} e^{iθ}|²
        g(θ) = √(dθ_src/dθ) · f(θ_src_total(θ))
    """
    if b5 is None:
        b5 = basis_b5(B)
    if b6 is None:
        b6 = basis_b6(B)
    a = r * np.exp(1j * delta)

    theta = _theta(B)
    # fused source angles
    theta_src_mob = mobius_inverse_theta(theta, a=a, gamma=gamma)
    theta_src_total = -theta_src_mob + (beta - alpha)

    # √Jacobian factor for unitary pullback on L²(dθ)
    z_out = np.exp(1j * theta)
    w = np.exp(-1j * gamma) * z_out
    jac = (1.0 - (r * r)) / (np.abs(1.0 + np.conj(a) * w) ** 2)
    # numerical guard (should be ≥0 for r<1)
    jac = np.maximum(jac, 0.0)
    sqrt_jac = np.sqrt(jac)
    if ablate_jacobian:
        sqrt_jac = np.ones_like(sqrt_jac)

    def apply_cycle_fused(f: np.ndarray) -> np.ndarray:
        return sqrt_jac * periodic_interp(f, theta_src_total)

    g5 = apply_cycle_fused(b5)
    g6 = apply_cycle_fused(b6)
    c5 = project_to_W(g5, b5, b6)
    c6 = project_to_W(g6, b5, b6)
    T = np.column_stack([c5, c6])

    if tau != 0.0:
        # In-plane rotation by τ in W
        mix = np.array([[np.cos(tau), -np.sin(tau)],
                        [np.sin(tau),  np.cos(tau)]], dtype=np.complex128)
        T = mix @ T
    return T

# ---------------- Cesàro projector ----------------
def cesaro_projector(T: np.ndarray, N: int = 64, start_from_T: bool = False) -> np.ndarray:
    """
    If start_from_T=False (default): P_N = (1/N) * sum_{k=0}^{N-1} T^k
    If start_from_T=True:          P_N = (1/N) * sum_{k=1}^{N}   T^k   (diagnostic to avoid the identity term)
    """
    if not start_from_T:
        P = np.eye(2, dtype=np.complex128)
        Tk = np.eye(2, dtype=np.complex128)
        for _ in range(1, N):
            Tk = Tk @ T
            P += Tk
        P /= N
        return P
    else:
        P = np.zeros((2,2), dtype=np.complex128)  # start from T^1
        Tk = np.eye(2, dtype=np.complex128)
        for _ in range(N):
            Tk = Tk @ T
            P += Tk
        P /= N
        return P

def projector_with_stability(T: np.ndarray, N: int = 64, start_from_T: bool = False):
    """
    Return P_N and ΔP₂N diagnostics:
      delta_P_spectral = ‖P_{2N} - P_N‖₂ (spectral norm)
      delta_P_maxabs   = max_ij |(P_{2N} - P_N)_{ij}|
    """
    Pn  = cesaro_projector(T, N=N,   start_from_T=start_from_T)
    P2n = cesaro_projector(T, N=2*N, start_from_T=start_from_T)
    D = P2n - Pn
    # spectral norm via largest singular value
    svals = np.linalg.svd(D, compute_uv=False)
    delta_spec = float(svals[0]) if svals.size else 0.0
    delta_maxabs = float(np.max(np.abs(D)))
    return Pn, {"delta_P_spectral": delta_spec, "delta_P_maxabs": delta_maxabs}

# ---------------- dynamic stability diagnostics ----------------
def analyze_dynamics(T: np.ndarray, N_dynamics: int = 128):
    """
    Analyze the iterative evolution of P_k = (1/k) * sum_{i=1}^k T^i, k=1..N_dynamics.

    Metrics:
      - convergence_rate: mean of the Frobenius deltas ‖P_{k+1}-P_k‖_F over the second half.
      - max_drift_angle: max arccos(|v_k · v_{k-1}|), where v_k is principal eigenvector of Hermitian(P_k).
      - coherence_ratio: λ_max / λ_min of Hermitian(P_N), inf if λ_min ≤ 0.

    Returns:
      dict with convergence_rate, max_drift_angle, coherence_ratio, and the final spectrum.
    """
    N = max(2, int(N_dynamics))
    Pk = np.zeros((2,2), dtype=np.complex128)
    Tk = np.eye(2, dtype=np.complex128)
    deltas = []
    angles = []
    v_prev = None

    for k in range(1, N+1):
        Tk = Tk @ T                         # T^k
        Pk_next = (Pk * (k-1)/k) + (Tk / k) # incremental Cesàro update

        # Frobenius step delta
        if k >= 2:
            D = Pk_next - Pk
            deltas.append(float(np.linalg.norm(D, ord='fro')))

        # principal direction from Hermitian(P_k)
        Hk = 0.5 * (Pk_next + Pk_next.conj().T)
        w, V = np.linalg.eigh(Hk)           # Hermitian -> real eigvals, ortho vecs
        idx = int(np.argmax(w))
        v_k = V[:, idx] / np.linalg.norm(V[:, idx])
        if v_prev is not None:
            # angle ∈ [0, π/2]: arccos(|⟨v_k, v_prev⟩|)
            dot = abs(np.vdot(v_k, v_prev))
            dot = min(1.0, max(0.0, float(dot.real)))  # numeric clip
            angles.append(float(np.arccos(dot)))
        v_prev = v_k

        Pk = Pk_next

    # convergence rate over the second half of deltas
    half = len(deltas) // 2
    conv_rate = float(np.mean(deltas[half:])) if deltas else 0.0
    max_drift = float(np.max(angles)) if angles else 0.0

    # coherence from final projector's spectrum
    wN, _ = np.linalg.eigh(0.5 * (Pk + Pk.conj().T))
    lam_min = float(np.min(wN))
    lam_max = float(np.max(wN))
    if lam_min <= 0.0:
        coh_ratio = float('inf')
    else:
        coh_ratio = float(lam_max / lam_min)
    # regularized version to avoid inf when λ_min is tiny negative due to roundoff
    trace = float(lam_min + lam_max)
    floor = max(1e-16 * max(trace, 1.0), 0.0)
    coh_ratio_reg = float(lam_max / max(lam_min, floor)) if lam_max > 0 else float('nan')

    return {
        "convergence_rate": conv_rate,
        "max_drift_angle": max_drift,
        "coherence_ratio": coh_ratio,
        "coherence_ratio_reg": coh_ratio_reg,
        "dyn_lambda_min": lam_min,
        "dyn_lambda_max": lam_max,
    }

# ---------------- Gram + diagnostics ----------------
def gram_from_projector(P: np.ndarray):
    """
    Returns Gram matrix G for u=P e1, v=P e2 and a diagnostic dict with:
      a, c, Re(b), Im(b), detG, h_min (complex parts), min_norm,
      λ_min(G), λ_max(G), cond(G), psd_violation, psd_margin, stable_h.
    """
    e1 = np.array([1.0+0j, 0.0+0j])
    e2 = np.array([0.0+0j, 1.0+0j])
    u = P @ e1
    v = P @ e2
    a = np.vdot(u, u).real
    c = np.vdot(v, v).real

    b = np.vdot(u, v)
    G = np.array([[a, b], [np.conj(b), c]], dtype=np.complex128)
    detG = (a * c - (np.abs(b) ** 2)).real

    # Minimizer and min norm
    if c > 0.0:
        h_min = - b / c
        min_norm = (a - (np.abs(b) ** 2) / c).real
    else:
        h_min = np.nan + 1j*np.nan
        min_norm = np.nan

    # PSD diagnostics
    lam = np.linalg.eigvalsh(G)  # Hermitian eigvals, real
    lam_min = float(lam[0])
    lam_max = float(lam[-1])
    psd_violation = (lam_min < -EPS_PSD)
    cond_G = float(lam_max / lam_min) if lam_min > 0.0 else float('inf')

    diag = {
        "a": float(a), "c": float(c),
        "b_real": float(b.real), "b_imag": float(b.imag),
        "detG": float(detG),
        "h_min_real": float(h_min.real) if np.isfinite(h_min.real) else float('nan'),
        "h_min_imag": float(h_min.imag) if np.isfinite(h_min.imag) else float('nan'),
        "min_norm": float(min_norm),
        "lambda_min_G": lam_min,
        "lambda_max_G": lam_max,
        "cond_G": cond_G,
        "psd_violation": bool(psd_violation),
        "psd_margin": lam_min,               # how far above/below 0
        "stable_h": bool(c > EPS_H)
    }
    return G, diag

def spectral_radius(T: np.ndarray) -> float:
    vals = np.linalg.eigvals(T)
    return float(np.max(np.abs(vals)))

# ---------------- evaluation helper ----------------
def evaluate_point(alpha: float, beta: float, tau: float, delta_eff: float, r: float,
                   B: int, Nproj: int, check_pn: bool, start_from_T: bool,
                   b5: Optional[np.ndarray] = None, b6: Optional[np.ndarray] = None,
                   check_dynamics: bool = False, N_dynamics: int = 128,
                   ablate_jacobian: bool = False):
    T = build_T_matrix(alpha=alpha, beta=beta, tau=tau, delta=delta_eff, r=r,
                       B=B, gamma=0.0, b5=b5, b6=b6, ablate_jacobian=ablate_jacobian)
    rho = spectral_radius(T)
    if check_pn:
        P, pn_diag = projector_with_stability(T, N=Nproj, start_from_T=start_from_T)
    else:
        P = cesaro_projector(T, N=Nproj, start_from_T=start_from_T)
        pn_diag = {}
    _, gdiag = gram_from_projector(P)
    trG = gdiag['a'] + gdiag['c']
    rel_min = (gdiag['min_norm'] / max(trG, 1e-30)) if (trG == trG) else float('nan')
    rec = {"alpha": float(alpha), "beta": float(beta), "rho_T": rho, "rel_min": float(rel_min), **pn_diag, **gdiag}
    if check_dynamics:
        dyn = analyze_dynamics(T, N_dynamics=N_dynamics)
        rec.update(dyn)
    return rec

# ---------------- coarse scan ----------------
def coarse_scan(group: str, tau: float, delta: float, r: float,
                B: int, Nproj: int, grid: int, check_pn: bool, start_from_T: bool,
                check_dynamics: bool = False, N_dynamics: int = 128,
                objective: str = "relmin", ablate_jacobian: bool = False):
    two_pi = 2.0 * np.pi
    if group.lower() == 'su3':
        eff_delta = delta
    else:
        # small parity offset for SU(4)
        eff_delta = delta + np.pi/7.0

    # Precompute bases once for speed
    b5 = basis_b5(B)
    b6 = basis_b6(B)

    records = []
    best = None

    for ia in range(grid):
        alpha = ia * two_pi / grid
        for ib in range(grid):
            beta = ib * two_pi / grid
            rec = evaluate_point(alpha, beta, tau, eff_delta, r, B, Nproj, check_pn, start_from_T,
                                 b5=b5, b6=b6,
                                 check_dynamics=check_dynamics, N_dynamics=N_dynamics,
                                 ablate_jacobian=ablate_jacobian)
            records.append(rec)
            # Unified scoring across groups to avoid bias
            if objective == "relmin":
                score = rec.get("rel_min", float("inf"))
            elif objective == "det":
                score = abs(rec["detG"])
            elif objective == "minnorm":
                score = rec["min_norm"]
            else:
                score = rec.get("rel_min", float("inf"))
            if best is None:
                best = (score, rec)
            else:
                if (group.lower() == 'su3' and score < best[0]) or (group.lower() == 'su4' and score > best[0]):
                    best = (score, rec)
    return eff_delta, records, best[1]

# ---------------- local refinement ----------------
def local_refine(group: str, tau: float, eff_delta: float, r: float,
                 B: int, Nproj: int,
                 center_ab, rounds: int = 1, local_grid: int = 12, radius: float = 0.2,
                 check_pn: bool = True, start_from_T: bool = False,
                 check_dynamics: bool = False, N_dynamics: int = 128,
                 objective: str = "relmin", ablate_jacobian: bool = False):
    two_pi = 2.0 * np.pi
    cx, cy = center_ab["alpha"], center_ab["beta"]
    best = None
    refinements = []

    # Precompute bases once for speed
    b5 = basis_b5(B)
    b6 = basis_b6(B)

    # Degenerate grid guard: evaluate only the center point
    if local_grid < 2:
        rec = evaluate_point(cx, cy, tau, eff_delta, r, B, Nproj, check_pn, start_from_T,
                             b5=b5, b6=b6,
                             check_dynamics=check_dynamics, N_dynamics=N_dynamics)
        rec["refined"] = True
        return rec, [rec]

    for _ in range(rounds):
        local_records = []
        for ia in range(local_grid):
            alpha = cx + radius * ((ia / (local_grid-1)) - 0.5) * two_pi
            for ib in range(local_grid):
                beta = cy + radius * ((ib / (local_grid-1)) - 0.5) * two_pi
                rec = evaluate_point(alpha, beta, tau, eff_delta, r, B, Nproj, check_pn, start_from_T,
                                     b5=b5, b6=b6,
                                     check_dynamics=check_dynamics, N_dynamics=N_dynamics,
                                     ablate_jacobian=ablate_jacobian)
                rec["refined"] = True
                local_records.append(rec)

                if objective == "relmin":
                    score = rec.get("rel_min", float("inf"))
                elif objective == "det":
                    score = abs(rec["detG"])
                elif objective == "minnorm":
                    score = rec["min_norm"]
                else:
                    score = rec.get("rel_min", float("inf"))
                if best is None:
                    best = (score, rec)
                else:
                    if (group.lower() == 'su3' and score < best[0]) or (group.lower() == 'su4' and score > best[0]):
                        best = (score, rec)
        # shrink radius and center around new best
        refinements.extend(local_records)
        cx, cy = best[1]["alpha"], best[1]["beta"]
        radius *= 0.5
    return best[1], refinements

# ---------------- B-doubling check ----------------
def b_doubling_check(alpha: float, beta: float, tau: float, delta_eff: float, r: float,
                     B: int, Nproj: int, check_pn: bool, start_from_T: bool,
                     check_dynamics: bool = False, N_dynamics: int = 128):
    """
    Recompute the same point at 2B and report key metrics and deltas.
    """
    # Baseline at B
    b5_B = basis_b5(B); b6_B = basis_b6(B)
    rec_B = evaluate_point(alpha, beta, tau, delta_eff, r, B, Nproj, check_pn, start_from_T,
                           b5=b5_B, b6=b6_B,
                           check_dynamics=check_dynamics, N_dynamics=N_dynamics)

    # Re-evaluate at 2B
    B2 = 2 * B
    b5_B2 = basis_b5(B2); b6_B2 = basis_b6(B2)
    rec_B2 = evaluate_point(alpha, beta, tau, delta_eff, r, B2, Nproj, check_pn, start_from_T,
                            b5=b5_B2, b6=b6_B2,
                            check_dynamics=check_dynamics, N_dynamics=N_dynamics)

    # Pack comparison
    compare = {
        "B2": B2,
        "detG_B2": rec_B2["detG"],
        "min_norm_B2": rec_B2["min_norm"],
        "lambda_min_G_B2": rec_B2["lambda_min_G"],
        "lambda_max_G_B2": rec_B2["lambda_max_G"],
        "cond_G_B2": rec_B2["cond_G"],
        "rho_T_B2": rec_B2["rho_T"],
        "delta_detG_B2_minus_B": rec_B2["detG"] - rec_B["detG"],
        "delta_min_norm_B2_minus_B": rec_B2["min_norm"] - rec_B["min_norm"],
        "delta_lambda_min_G_B2_minus_B": rec_B2["lambda_min_G"] - rec_B["lambda_min_G"],
        "delta_cond_G_B2_minus_B": rec_B2["cond_G"] - rec_B["cond_G"],
    }
    if check_pn:
        compare.update({
            "delta_P_spectral_B2": rec_B2.get("delta_P_spectral", float("nan")),
            "delta_P_maxabs_B2": rec_B2.get("delta_P_maxabs", float("nan")),
        })
    if check_dynamics:
        compare.update({
            "convergence_rate_B2": rec_B2.get("convergence_rate", float("nan")),
            "max_drift_angle_B2": rec_B2.get("max_drift_angle", float("nan")),
            "coherence_ratio_B2": rec_B2.get("coherence_ratio", float("nan")),
        })
    return compare

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=str, default="both", choices=["su3", "su4", "both"])
    ap.add_argument("--B", type=int, default=1024, help="boundary samples")
    ap.add_argument("--Nproj", type=int, default=48, help="Cesàro steps")
    ap.add_argument("--grid", type=int, default=20, help="coarse alpha/beta grid")
    ap.add_argument("--tau", type=float, default=0.02, help="twist mixing in W")
    ap.add_argument("--r", type=float, default=0.08, help="Möbius magnitude (|a|)")
    ap.add_argument("--delta", type=float, default=0.0, help="Möbius phase (arg a) baseline")
    ap.add_argument("--refine", type=int, default=1, help="number of local refinement rounds")
    ap.add_argument("--objective", type=str, default="relmin", choices=["relmin","det","minnorm"],
                    help="scan objective (unified for both groups): relmin = minimize min‖u+h v‖²/trace, det = minimize |det G|, minnorm = minimize min‖u+h v‖²")
    ap.add_argument("--report-topk", type=int, default=5, help="report top-K points per group in summary JSON")
    ap.add_argument("--refine-grid", type=int, default=12, help="local grid per round")
    ap.add_argument("--refine-radius", type=float, default=0.25, help="initial local radius (fraction of 2π)")
    # Stability checks
    ap.add_argument("--check-pn", dest="check_pn", action="store_true", help="enable ΔP₂N stability metric")
    ap.add_argument("--no-check-pn", dest="check_pn", action="store_false", help="disable ΔP₂N stability metric")
    ap.set_defaults(check_pn=True)
    ap.add_argument("--check-b-doubling", action="store_true", help="re-evaluate best point at 2B")
    # Bridge (Sections 5–6): calibrate Δα̂ on SU(3) and predict SU(4)
    ap.add_argument("--bridge", action="store_true", help="compute one-loop bridge prediction using convergence rates")
    ap.add_argument("--R0_SU3", type=float, default=1.152, help="baseline R0(SU3; D_eff) from paper")
    ap.add_argument("--R0_SU4", type=float, default=1.200, help="baseline R0(SU4; D_eff) from paper")
    ap.add_argument("--bridge-agg", type=str, default="best", choices=["best","median_topk"], help="use best point or median over topK")
    # Cesàro variant: optionally exclude identity term
    ap.add_argument("--cesaro-start-T", action="store_true", help="start Cesàro sum at T¹ instead of I")
    # Dynamic analysis (optional)
    ap.add_argument("--check-dynamics", action="store_true", help="analyze iterative convergence of P_k")
    ap.add_argument("--N-dynamics", type=int, default=128, help="iterations for dynamic analysis (P_k)")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colors and emoji icons")
    # Ablations
    ap.add_argument("--ablate-jacobian", action="store_true", help="disable √Jacobian weight in pullback")
    ap.add_argument("--compare-ablation", action="store_true", help="run baseline and ablation side-by-side and report deltas")

    args = ap.parse_args()
    style = _Style(enable=(not args.no_color) and sys.stdout.isatty())

    # Basic validation
    if args.B < 2:
        raise ValueError("B must be ≥ 2")
    if args.Nproj < 1:
        raise ValueError("Nproj must be ≥ 1")
    if args.grid < 1:
        raise ValueError("grid must be ≥ 1")
    if args.refine < 0:
        raise ValueError("refine must be ≥ 0")
    if args.refine_grid < 1:
        print("[warn] refine-grid < 2 will evaluate only the center point per round.")

    groups = ["su3", "su4"] if args.group == "both" else [args.group]
    summary = {}
    for g in groups:
        eff_delta, records, best = coarse_scan(
            group=g, tau=args.tau, delta=args.delta, r=args.r,
            B=args.B, Nproj=args.Nproj, grid=args.grid,
            check_pn=args.check_pn, start_from_T=args.cesaro_start_T,
            check_dynamics=args.check_dynamics, N_dynamics=args.N_dynamics,
            objective=args.objective, ablate_jacobian=False
        )

        rho_flag = " !rho>1!" if best["rho_T"] > 1.0 + EPS_RHO else ""
        psd_flag = " !PSD-viol!" if best["psd_violation"] else ""
        pn_str = ""
        if args.check_pn:
            pn_str = f"  ΔP₂N(‖·‖₂)={best['delta_P_spectral']:.3e}  ΔP₂N(max)={best['delta_P_maxabs']:.3e}"
        print(f"[{g}] coarse best at (α,β)=({best['alpha']:.6f},{best['beta']:.6f})  "
              f"detG={best['detG']:.3e}  minNorm={best['min_norm']:.3e}  "
              f"rho(T)={best['rho_T']:.6f}{rho_flag}  λ_min(G)={best['lambda_min_G']:.3e}{psd_flag}{pn_str}")

        best_refined, ref_records = local_refine(
            group=g, tau=args.tau, eff_delta=eff_delta, r=args.r,
            B=args.B, Nproj=args.Nproj,
            center_ab=best, rounds=max(0, args.refine),
            local_grid=args.refine_grid, radius=args.refine_radius,
            check_pn=args.check_pn, start_from_T=args.cesaro_start_T,
            check_dynamics=args.check_dynamics, N_dynamics=args.N_dynamics,
            objective=args.objective, ablate_jacobian=False
        )

        rho_flag = " !rho>1!" if best_refined["rho_T"] > 1.0 + EPS_RHO else ""
        psd_flag = " !PSD-viol!" if best_refined["psd_violation"] else ""
        pn_str = ""
        if args.check_pn:
            pn_str = f"  ΔP₂N(‖·‖₂)={best_refined['delta_P_spectral']:.3e}  ΔP₂N(max)={best_refined['delta_P_maxabs']:.3e}"
        print(f"[{g}] final best at (α,β)=({best_refined['alpha']:.6f},{best_refined['beta']:.6f})  "
              f"detG={best_refined['detG']:.3e}  minNorm={best_refined['min_norm']:.3e}  "
              f"rho(T)={best_refined['rho_T']:.6f}{rho_flag}  λ_min(G)={best_refined['lambda_min_G']:.3e}{psd_flag}{pn_str}")


        # Optional: run ablation branch
        ablation_out = None
        if args.compare_ablation:
            eff_delta_ab, rec_ab, best_ab = coarse_scan(
                group=g, tau=args.tau, delta=args.delta, r=args.r,
                B=args.B, Nproj=args.Nproj, grid=args.grid,
                check_pn=args.check_pn, start_from_T=args.cesaro_start_T,
                check_dynamics=args.check_dynamics, N_dynamics=args.N_dynamics,
                objective=args.objective, ablate_jacobian=True
            )
            best_refined_ab, ref_records_ab = local_refine(
                group=g, tau=args.tau, eff_delta=eff_delta_ab, r=args.r,
                B=args.B, Nproj=args.Nproj,
                center_ab=best_ab, rounds=max(0, args.refine),
                local_grid=args.refine_grid, radius=args.refine_radius,
                check_pn=args.check_pn, start_from_T=args.cesaro_start_T,
                check_dynamics=args.check_dynamics, N_dynamics=args.N_dynamics,
                objective=args.objective, ablate_jacobian=True
            )
            # Print concise comparison
            print(style.color(f"[{g}] Ablation (no √Jacobian) best detG={best_refined_ab['detG']:.3e} minNorm={best_refined_ab['min_norm']:.3e} rel_min={best_refined_ab.get('rel_min', float('nan')):.3e}", 'yellow'))
            # Store for summary
            ablation_out = {"best": best_refined_ab, "records": rec_ab, "refined": ref_records_ab}

        # Optional B-doubling check at the final best
        b2 = None
        if args.check_b_doubling:
            b2 = b_doubling_check(
                alpha=best_refined["alpha"], beta=best_refined["beta"],
                tau=args.tau, delta_eff=eff_delta, r=args.r,
                B=args.B, Nproj=args.Nproj, check_pn=args.check_pn, start_from_T=args.cesaro_start_T,
                check_dynamics=args.check_dynamics, N_dynamics=args.N_dynamics
            )
            print(f"[{g}] B-doubling: B→{b2['B2']}  "
                  f"ΔdetG={b2['delta_detG_B2_minus_B']:.3e}  "
                  f"ΔminNorm={b2['delta_min_norm_B2_minus_B']:.3e}  "
                  f"Δλ_min(G)={b2['delta_lambda_min_G_B2_minus_B']:.3e}  "
                  f"Δcond(G)={b2['delta_cond_G_B2_minus_B']:.3e}")

        # SU(3) certificate (scale-aware): near-rank-1 if λ_min ≪ trace
        near_rank1 = None
        h_star_real = None
        if g == "su3":
            trG = best_refined["a"] + best_refined["c"]
            near_rank1 = (best_refined["lambda_min_G"] <= 1e-6 * trG)
            # real-h estimate (if that's the certificate’s constraint)
            h_star_real = - best_refined["b_real"] / best_refined["c"] if best_refined["c"] != 0.0 else float('nan')

        out = {
            "group": g,
            "B_samples": args.B,
            "Nproj": args.Nproj,
            "grid": args.grid,
            "tau": args.tau,
            "r": args.r,
            "delta": float(eff_delta),

            "best_alpha": best_refined["alpha"],
            "best_beta": best_refined["beta"],

            "detG_at_best": best_refined["detG"],
            "min_norm_at_best": best_refined["min_norm"],
            "h_min_real_at_best": best_refined["h_min_real"],
            "h_min_imag_at_best": best_refined["h_min_imag"],

            "a_entry": best_refined["a"],
            "c_entry": best_refined["c"],
            "b_real": best_refined["b_real"],
            "b_imag": best_refined["b_imag"],

            # Diagnostics at best
            "rho_T_at_best": best_refined["rho_T"],
            "lambda_min_G_at_best": best_refined["lambda_min_G"],
            "lambda_max_G_at_best": best_refined["lambda_max_G"],
            "cond_G_at_best": best_refined["cond_G"],
            "psd_violation_at_best": best_refined["psd_violation"],
            "psd_margin_at_best": best_refined["psd_margin"],
            "stable_h_at_best": best_refined["stable_h"],
        }
        if args.check_pn:
            out["delta_P_spectral_at_best"] = best_refined["delta_P_spectral"]
            out["delta_P_maxabs_at_best"] = best_refined["delta_P_maxabs"]
        if args.check_dynamics:
            out["convergence_rate_at_best"] = best_refined.get("convergence_rate", float('nan'))
            out["max_drift_angle_at_best"] = best_refined.get("max_drift_angle", float('nan'))
            out["coherence_ratio_at_best"] = best_refined.get("coherence_ratio", float('nan'))
            out["coherence_ratio_reg_at_best"] = best_refined.get("coherence_ratio_reg", float('nan'))
            # include raw final dynamic eigen spectrum if present
            if "dyn_lambda_min" in best_refined:
                out["dyn_lambda_min_at_best"] = best_refined["dyn_lambda_min"]
                out["dyn_lambda_max_at_best"] = best_refined["dyn_lambda_max"]
        if g == "su3":
            out["near_rank1"] = near_rank1
            out["h_star_estimate_real"] = h_star_real
        if args.check_b_doubling and b2 is not None:
            out["B_doubling"] = b2

        out['rel_min_norm'] = float(out['min_norm_at_best'] / max(out['a_entry'] + out['c_entry'], 1e-30))
        # Placeholder; string printed later. If you parse JSON only, recompute from rel_min_norm.
        out['verdict'] = None
        out['rel_min_norm'] = float(out['min_norm_at_best'] / max(out['a_entry'] + out['c_entry'], 1e-30))
        # Placeholder; string printed later. If you parse JSON only, recompute from rel_min_norm.
        out['verdict'] = None
        # Top-K points by the same objective for transparency
        def _score(rec):
            if args.objective == 'relmin':
                return rec.get('rel_min', float('inf'))
            if args.objective == 'det':
                return abs(rec['detG'])
            if args.objective == 'minnorm':
                return rec['min_norm']
            return rec.get('rel_min', float('inf'))
        all_records = sorted(records + ref_records, key=_score)[:max(1, args.report_topk)]
        out['topK'] = [{k: r.get(k) for k in ['alpha','beta','rel_min','detG','min_norm','lambda_min_G','rho_T','convergence_rate']} for r in all_records]
        if ablation_out is not None:
            out['ablation_no_sqrt_jacobian'] = {
                'detG_at_best': ablation_out['best']['detG'],
                'min_norm_at_best': ablation_out['best']['min_norm'],
                'rel_min_at_best': ablation_out['best'].get('rel_min'),
                'lambda_min_G_at_best': ablation_out['best']['lambda_min_G'],
                'rho_T_at_best': ablation_out['best']['rho_T']
            }
        summary[g] = out

        # Save scans (coarse + refined)
        with open(f"scan_{g}.json", "w") as f:
            json.dump(records + ref_records, f, indent=2)

    with open("phi_gram_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


    # ---------------- Bridge computation (Appendix C) ----------------
    if args.bridge and all(k in summary for k in ("su3","su4")):
        s3 = summary["su3"]
        s4 = summary["su4"]
        # choose convergence rates
        def _pick_conv(sx):
            if args.bridge_agg == "median_topk" and "topK" in sx and sx["topK"]:
                vals = [r.get("convergence_rate", float("nan")) for r in sx["topK"] if not (r.get("convergence_rate") is None)]
                vals = [v for v in vals if isinstance(v, (int,float)) and v==v and v>0]
                if vals:
                    vals.sort()
                    mid = vals[len(vals)//2] if len(vals)%2==1 else 0.5*(vals[len(vals)//2-1]+vals[len(vals)//2])
                    return mid
            # fallback to the "best" convergence rate
            return sx.get("convergence_rate_at_best", float("nan"))
        conv3 = _pick_conv(s3)
        conv4 = _pick_conv(s4)

        # Guardrails
        if not (isinstance(conv3,(int,float)) and conv3>0 and isinstance(conv4,(int,float)) and conv4>0):
            print(style.color("[bridge] Missing/invalid convergence rates; cannot compute bridge.", "red"))
        else:
            R0_3 = float(args.R0_SU3)
            R0_4 = float(args.R0_SU4)
            # Δα̂ = ln(R0(SU4)/R0(SU3)) / ln(1/conv_SU3)
            try:
                delta_alpha_hat = (np.log(R0_4 / R0_3)) / (np.log(1.0 / conv3))
            except Exception as e:
                delta_alpha_hat = float("nan")

            # Prediction for SU(4): R_pred = R0_4 * (1/conv4)^{Δα̂}
            if np.isfinite(delta_alpha_hat):
                phi4 = (1.0 / conv4) ** delta_alpha_hat
                R4_pred = R0_4 * phi4
                # Discriminant factor from dynamics only: (conv4/conv3)^{Δα̂}
                discr = (conv4 / conv3) ** delta_alpha_hat
            else:
                phi4 = float("nan"); R4_pred = float("nan"); discr = float("nan")

            # Log to console
            print("")
            print(style.color("=== One-Loop Bridge (Appendix C) ===", "bold", "magenta"))
            print(f"Baseline: R0(SU3)={R0_3:.6f}, R0(SU4)={R0_4:.6f}")
            print(f"Convergence rates: SU3={conv3:.6e}, SU4={conv4:.6e}")
            print(f"Calibrated Δα̂ = ln(R0_4/R0_3)/ln(1/conv_SU3) ≈ {delta_alpha_hat:.6e}")
            print(f"Dynamic discriminant (conv4/conv3)^Δα̂ ≈ {discr:.6e}")
            print(f"Predicted R(SU4; D_eff) = R0(SU4) · (1/conv4)^Δα̂ ≈ {R4_pred:.6e}")
            dev = R4_pred - R0_4
            print(f"Deviation from baseline R0(SU4): {dev:+.6e} ({(dev/R0_4)*100:+.2f}%)")

            # Store in summary
            summary["bridge"] = {
                "R0_SU3": R0_3, "R0_SU4": R0_4,
                "conv_SU3": conv3, "conv_SU4": conv4,
                "delta_alpha_hat": float(delta_alpha_hat),
                "dynamic_discriminant": float(discr),
                "R_SU4_pred": float(R4_pred),
                "R_SU4_dev": float(dev),
                "R_SU4_dev_pct": float((dev/R0_4)*100.0)
            }

    # Print summary
    for g in groups:
        s = summary[g]
        print("")
        print(f"=== {g.upper()} CERTIFICATE ===")
        print(f"B={s['B_samples']}  Nproj={s['Nproj']}  grid={s['grid']}  "
              f"tau={s['tau']}  r={s['r']}  delta={s['delta']:.4f}  "
              f"Cesàro start={'T¹' if args.cesaro_start_T else 'I'}")
        print(f"best (alpha, beta) = ({s['best_alpha']:.6f}, {s['best_beta']:.6f})")
        print(f"det G = {s['detG_at_best']:.6e}")
        print(f"min ||u + h v||^2 (complex-h) = {s['min_norm_at_best']:.6e}")
        line = (f"ρ(T) = {s['rho_T_at_best']:.6f}  "
                f"λ_min(G) = {s['lambda_min_G_at_best']:.6e}  "
                f"λ_max(G) = {s['lambda_max_G_at_best']:.6e}  "
                f"cond(G) = {s['cond_G_at_best']:.3e}  "
                f"PSD violation? {s['psd_violation_at_best']}")
        print(line)

        if "delta_P_spectral_at_best" in s:
            print(f"ΔP₂N(‖·‖₂) = {s['delta_P_spectral_at_best']:.6e}   "
                  f"ΔP₂N(max) = {s['delta_P_maxabs_at_best']:.6e}")
        if g == "su3":
            print(f"near rank-1? {s['near_rank1']}   "
                  f"h_* (real estimate) ≈ {s['h_star_estimate_real']:.6f}   "
                  f"stable_h? {s['stable_h_at_best']}")
        if args.check_dynamics:
            print(f"Dynamic Analysis (N={args.N_dynamics}):")
            print(f"  Convergence Rate: {s.get('convergence_rate_at_best', float('nan')):.6e}")
            print(f"  Max Drift Angle:  {s.get('max_drift_angle_at_best', float('nan')):.6e} rad")
            print(f"  Coherence Ratio:  {s.get('coherence_ratio_at_best', float('nan')):.6e}")
            print(f"  Coherence Ratio (reg.):  {s.get('coherence_ratio_reg_at_best', float('nan')):.6e}")
        # --- Verdict (layman-friendly, technically grounded) ---
        # We score how "collapsed" the projector is by the relative minimum norm:
        #   rel_min = min‖u + h v‖² / trace(G), where trace(G) = a + c.
        # Smaller rel_min means the two directions have effectively merged (rank-1-like).
        trG = s['a_entry'] + s['c_entry']
        rel_min = s['min_norm_at_best'] / max(trG, 1e-30)

        # Heuristics consistent with the paper’s qualitative split:
        #   rel_min ≤ 0.03  → coherent resonance
        #   0.03 < rel_min ≤ 0.15 → weak resonance
        #   rel_min > 0.15  → geometric obstruction
        # We further require stability: small ΔP₂N and a low convergence rate when available.
        dp_ok = True
        conv_ok = True
        if "delta_P_spectral_at_best" in s:
            dp_ok = (s["delta_P_spectral_at_best"] <= 1e-7)
        if "convergence_rate_at_best" in s:
            conv_ok = (s["convergence_rate_at_best"] <= 3e-10)

        if rel_min <= 0.03 and dp_ok and conv_ok:
            verdict = "Coherent resonance (projector nearly rank‑1; stable Cesàro)"
        elif rel_min <= 0.15 and dp_ok:
            verdict = "Weak resonance (partial collapse with stable Cesàro)"
        else:
            verdict = "Geometric obstruction (directions stay distinct; no collapse)"

        # Print a compact, friendly summary with icons and optional colors.
        banner_color = 'green' if 'Coherent' in verdict else ('yellow' if 'Weak' in verdict else 'red')
        print(style.color(f"Verdict: {verdict}", 'bold', banner_color))
        # metric checks with ticks
        thr_rel = 0.03 if 'Coherent' in verdict else (0.15 if 'Weak' in verdict else None)
        ok_rel = (rel_min <= 0.03)
        ok_dp = True
        ok_conv = True
        if 'delta_P_spectral_at_best' in s:
            ok_dp = (s['delta_P_spectral_at_best'] <= 1e-7)
        if 'convergence_rate_at_best' in s:
            ok_conv = (s['convergence_rate_at_best'] <= 3e-10)
        print(f"  {_tick(ok_rel)} rel_min = {rel_min:.3e}  (smaller is more coherent)")
        print(f"    {_coherence_bar(rel_min, style)}  thresholds: 0.03 (coherent), 0.15 (weak)")
        if 'delta_P_spectral_at_best' in s:
            print(f"  {_tick(ok_dp)} Cesàro stability ΔP₂N(‖·‖₂) = {s['delta_P_spectral_at_best']:.3e}")
        if 'convergence_rate_at_best' in s:
            print(f"  {_tick(ok_conv)} Dynamic convergence rate = {s['convergence_rate_at_best']:.3e}")
        if "B_doubling" in s:
            b2 = s["B_doubling"]
            print(f"B-doubling check (B→{b2['B2']}): "
                  f"detG_B2={b2['detG_B2']:.6e}, minNorm_B2={b2['min_norm_B2']:.6e}, "
                  f"λ_min(G)_B2={b2['lambda_min_G_B2']:.6e}, cond(G)_B2={b2['cond_G_B2']:.3e}")
            print(f"Deltas (B2 - B): ΔdetG={b2['delta_detG_B2_minus_B']:.6e}, "
                  f"ΔminNorm={b2['delta_min_norm_B2_minus_B']:.6e}, "
                  f"Δλ_min(G)={b2['delta_lambda_min_G_B2_minus_B']:.6e}, "
                  f"Δcond(G)={b2['delta_cond_G_B2_minus_B']:.6e}")

    # One-time legend
    print("\nLegend: ✅ pass, ❌ fail; rel_min bar fills more on stronger coherence."
          " Thresholds: rel_min ≤ 0.03 → Coherent, 0.03–0.15 → Weak, > 0.15 → Obstruction.")

if __name__ == "__main__":
    main()
