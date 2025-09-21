# PHI‑GRAM: Technical README

**Scope.** End‑to‑end validation of the geometric filter (Sections 2–4) and the one‑loop bridge (Sections 5–6) for SU(3) vs SU(4), with ablations, stability checks, and transparent scans. This README documents the **algorithm**, **parameters**, **metrics**, **CLI**, and **reproducibility**.

---

## 0) Quickstart (research defaults)

```bash
python phi_gram_ref.py --cesaro-start-T --check-dynamics --group both --bridge --R0_SU3 1.152 --R0_SU4 1.20
```

For robustness:
```bash
python phi_gram_ref.py --cesaro-start-T --check-dynamics --group both   --grid 28 --refine 2 --refine-grid 16 --B 4096 --Nproj 96   --bridge --bridge-agg median_topk
```

Ablation comparison:
```bash
python phi_gram_ref.py --cesaro-start-T --check-dynamics --group both --compare-ablation
```

---

## 1) Construction (operator `T` on `W = span{b₅, b₆}`)

- Boundary samples: `θ_j = 2π j / B`, `j = 0..B−1`.
- Basis functions: `b₅(θ) = e^{i·5θ}`, `b₆(θ) = e^{i·6θ}`.
- Möbius map: `z ↦ e^{iγ} (z − a) / (1 − ar a z)` with `a = r·e^{i·δ}`, `0 ≤ r < 1`.
- Inverse boundary angle: `θ_in = arg( (w + a) / (1 + ar a w) )`, `w = e^{−iγ} e^{iθ}`.
- Fused source angle with parity flip and co‑shift: `θ_src_total(θ) = −θ_in + (β − α)`.
- **Unitary pullback weight**: `√(dθ_src/dθ) = √( (1 − |a|²) / |1 + ar a w|² )`.
- Twisted 2×2 operator: evaluate pullback on `b₅, b₆`, project to `W`, then apply in‑plane rotation by `τ`:
  `T = R(τ) · Proj_W ∘ Pullback`.

**Ablation**: `--ablate-jacobian` sets the weight to 1 (drop `√Jacobian`).

---

## 2) Cesàro projector and stability

- Cesàro: `P_N = (1/N) ∑_{k=0}^{N−1} T^k` (default) or `P_N = (1/N) ∑_{k=1}^{N} T^k` with `--cesaro-start-T`.
- Stability diagnostic: compute `P_{2N}`; report
  - `ΔP₂N(‖·‖₂) = ‖P_{2N} − P_N‖₂` (spectral norm via σ₁),
  - `ΔP₂N(max) = max_ij |(P_{2N} − P_N)_{ij}|`.
- Spectral radius: `ρ(T) = max |λ_i(T)|` (guardrail only).

---

## 3) Gram matrix and scalar observables

Let `u = P e₁`, `v = P e₂`, with `e₁ = (1,0), e₂ = (0,1)` in the `{b₅,b₆}` basis.

- Gram: `G = [[⟨u,u⟩, ⟨u,v⟩], [⟨v,u⟩, ⟨v,v⟩]]` (Hermitian 2×2).
- Determinant: `det G = ⟨u,u⟩⟨v,v⟩ − |⟨u,v⟩|²`.
- Complex minimizer: `h_min = −⟨u,v⟩ / ⟨v,v⟩` when `⟨v,v⟩ > 0`.
- Minimum norm: `min‖u + h v‖² = ⟨u,u⟩ − |⟨u,v⟩|² / ⟨v,v⟩`.
- Trace: `tr(G) = ⟨u,u⟩ + ⟨v,v⟩`.
- **Relative minimum** (collapse score): `rel_min = min‖u + h v‖² / tr(G)`.

PSD check uses `eigvalsh(G)`; we report `λ_min(G), λ_max(G), cond(G)`. A “regularized” coherence ratio uses a trace‑scaled floor to avoid `∞` from tiny negative `λ_min` due to finite‑N rounding.

---

## 4) Dynamics (signature)

We monitor `P_k = (1/k) ∑_{i=1}^k T^i` for `k = 1..N_dyn` and compute:
- `convergence_rate` = mean of `‖P_{k+1} − P_k‖_F` over the second half of iterations,
- `max_drift_angle` for principal eigenvector of `Hermitian(P_k)`,
- `coherence_ratio` and a regularized variant from the final `Hermitian(P_N)` spectrum.

SU(3) exhibits **slower** convergence (smaller rate) than SU(4) when coherent collapse occurs.

---

## 5) Scanning and selection

- Coarse grid over `(α, β)` with cache for `b₅, b₆` and periodic interpolation.
- Local refinement around the current best with shrinking radius.
- **Unified objective** (no per‑group bias): `--objective relmin|det|minnorm` (default `relmin`).
- Transparency: `--report-topk K` stores the top‑K records (by the chosen objective) into the summary JSON.

---

## 6) Verdict (rule, thresholds)

The printed verdict uses `rel_min` and stability:
- `rel_min ≤ 0.03` and stable ⇒ **Coherent resonance**.
- `0.03 < rel_min ≤ 0.15` and stable ⇒ **Weak resonance**.
- otherwise ⇒ **Geometric obstruction**.

This is a presentational layer; raw metrics are preserved in JSON.

---

## 7) Bridge (Appendix‑style)
We implement a minimal bridge consistent with the paper’s derivation:

- Baseline: `R0(SU3; D_eff)`, `R0(SU4; D_eff)` supplied via `--R0_SU3`, `--R0_SU4`.
- Dynamic factor: `Φ(G) = (1 / convergence_rate(G))^{Δα̂}`.
- Calibrate on SU(3):  
  `Δα̂ = ln( R0(SU4)/R0(SU3) ) / ln( 1 / convergence_rate_SU3 )`.
- Predict SU(4):  
  `R_pred(SU4; D_eff) = R0(SU4) · (1 / convergence_rate_SU4)^{Δα̂}`.
- Report the discriminant `(conv4/conv3)^{Δα̂}` and deviation `R_pred − R0(SU4)`.
- Aggregation: `--bridge-agg best|median_topk` (median gives robustness against grid noise).

**Outputs** are stored under `"bridge"` in `phi_gram_summary.json`.

---

## 8) Ablations

- `--ablate-jacobian`: drop the `√(dθ_src/dθ)` compensation.
- `--compare-ablation`: run **baseline** and **ablation** side‑by‑side; store compact ablation results under `"ablation_no_sqrt_jacobian"` in the summary JSON.
- (Extendable) Additional toggles can be added for parity flip, twist `τ`, or Cesàro start.

Expected: ablation degrades coherence and/or dynamics relative to baseline; the geometric split becomes less pronounced.

---

## 9) Outputs

- `scan_su3.json`, `scan_su4.json`: per‑point records with `alpha, beta, rel_min, detG, min_norm, lambda_min_G, rho_T` and, when dynamics enabled, `convergence_rate`.
- `phi_gram_summary.json`: per‑group best, diagnostics, optional `"bridge"`, optional `"ablation_no_sqrt_jacobian"`, and `"topK"` transparency list.
- Console: certificates with numerical metrics, a verdict, checkmarks, and a coherence bar.

---

## 10) CLI (selected)

```
--group su3|su4|both
--B <int>                # boundary samples (default 1024; try 4096 for robustness)
--Nproj <int>            # Cesàro steps (default 48; try 96 for robustness)
--grid <int>             # coarse α/β grid size
--refine <int>           # local refinement rounds
--refine-grid <int>      # local grid per round
--refine-radius <float>  # initial local radius (fraction of 2π)

--tau <float>            # in-plane twist in W (default 0.02)
--r <float>, --delta     # Möbius parameter a = r·e^{i·δ}
--cesaro-start-T         # start Cesàro sum at T¹ instead of I
--check-dynamics         # compute dynamic signatures (recommended)
--N-dynamics <int>

--objective relmin|det|minnorm
--report-topk <int>
--check-b-doubling
--ablate-jacobian
--compare-ablation

--bridge --R0_SU3 <float> --R0_SU4 <float> [--bridge-agg best|median_topk]
--no-color
```

---

## 11) Complexity & practical settings

- Each evaluation builds one `2×2` matrix `T` from `B` samples (vectorized), then computes a handful of 2×2 operations. The scan cost is dominated by interpolation and the Cesàro loop (`O(B·Nproj)` per point).
- Recommended robustness set: `B=4096`, `Nproj=96`, `grid≥28`, `refine=2`, `refine-grid=16`.
- Dynamics: `N_dynamics=128` is typically sufficient.

---

## 12) Reproducibility & audits

- No randomness; outputs are deterministic given flags.
- Transparency: all raw scan records and the selected top‑K are saved. The **verdict** is derived **only** from `rel_min` and stability thresholds; you can recompute it offline from JSON (`rel_min_norm` is included).
- B‑doubling checks (`--check-b-doubling`) confirm resolution stability.
- The ablation pathway provides a falsifiable sanity check that the guardrails matter.

---

## 13) Notes / extensions

- Alternate bases: trivially extend to `b_k, b_{k+1}` by changing `basis_b5/b6`.
- Multiple twists or Möbius phases: can be scanned with shell loops.
- PDF‑style tables/plots: JSON is structured for downstream notebooks.

