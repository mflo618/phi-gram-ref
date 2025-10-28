## 1) `README.md`

````md
# GRC Paper — Minimal Audience Pack

This mini-pack runs the *five checks* used in the paper:
1) Offset removal (SU4 @ (5,6): δ=0 vs δ=π/7)
2) γ-invariance (SU3 @ (5,6): γ ∈ {0, π/4, π/2, 3π/4, π})
3) Union-null thresholds (rank & Coxeter harmonic families; 10 scrambles each)
4) SU(2)–SU(6) suite: z-scores, accessibility, Spearman ρ (dynamic_z vs N)
5) Minimal blinded check (SU2–SU4)

**Assumption:** your scanner entrypoint exists at `python phi_gram_ref.py`.

## Quick start

```bash
# From the repo root where phi_gram_ref.py lives:
python run_grc_protocol.py --config config.grc.json
````

Artifacts land in `./grc_outputs/`:

* `offset_result.json` — SU4 δ=0 vs π/7 verdicts
* `gamma_scan_su3_5_6.json` — γ vs metrics/verdicts
* `null_thresholds.json` — derived thresholds + summary stats (for z-scores)
* `su_suite_results.json` — rows for SU(2)…SU(6) with z-scores, accessibility, Spearman ρ
* `blinded_min.json` — masked→unmasked 3-row check

## Requirements

* Python 3.9+
* Your repo’s scanner (`phi_gram_ref.py`) runnable via `python phi_gram_ref.py`
* Stdlib only (no extra packages)

## Notes

* Objective is fixed to **minimize rel_min** for all runs.
* If γ-invariance holds, we set γ=0 for everything else, and state that in Methods.
* The null uses **pre-declared harmonic families** (rank & Coxeter) with small scrambles; thresholds are 1% tails.

````

---

