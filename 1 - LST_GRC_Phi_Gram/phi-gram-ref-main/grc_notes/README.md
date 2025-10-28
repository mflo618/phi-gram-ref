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

### Script Roles: Discovery vs. Validation

The scientific process embodied in this repository is split between two key scripts: the "Discovery Engine" (`phi_gram_ref.py`) and the "Validation Engine" (`run_grc_protocol.py`).

#### 1. The Discovery Engine (`phi_gram_ref.py`)

This script is the core implementation of the theory.

-   **Implements the Geometric Filter:** It contains the direct mathematical model described in the paper.
-   **Finds the Key Result:** It is used to scan the parameter space and discover the selective resonance pattern.
-   **Source of Paper's Data:** The main results table in the paper is a direct report from the output of this script.

#### 2. The Validation Engine (`run_grc_protocol.py`)

This script serves as an **Automated Robustness and Validation Protocol**. It takes the discovery from the first script as a hypothesis and then attempts to falsify it through a series of automated checks to ensure the result is not a fragile artifact.

| Component in `run_grc_protocol.py` | Its Role in Validation | What It Proves |
| :----------------------------------- | :--------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `gate_offset` & `gate_delta_scan`    | **Parameter Invariance & Sensitivity Analysis**  | Checks if the core result (the verdict) is stable when a key parameter (`delta`) is changed. This demonstrates that the finding is not a result of "fine-tuning" or a lucky parameter choice.                                       |
| `gate_null`                          | **Statistical Significance & Threshold Calibration** | Generates a "null distribution" by running many scrambled, non-canonical cases. It then calculates what a "significant" result is based on a statistical confidence level (`alpha=0.01`), replacing simple hardcoded thresholds. |
| `gate_su_suite`                      | **Formal Protocol Execution**                    | Re-runs the main SU(3) and SU(4) tests but judges them against the more rigorous, statistically-derived thresholds from `gate_null`. This replaces an "eyeball" test with a formal statistical one.                                   |
| `gate_blinded_min`                   | **Mitigation of Experimenter Bias**              | Masks the identities of SU(3) and SU(4) (as `"A"` and `"B"`), ensuring that the result is judged purely on its numerical output, removing any possibility of conscious or unconscious bias in the analysis or code.                |

In summary, the validation engine's purpose is to provide additional evidence that the discovered pattern is **not an insignificant artifact**.
