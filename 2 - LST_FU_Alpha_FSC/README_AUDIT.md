# Audit Add-Ons — Read Me

This folder contains small, single-purpose scripts that let a reviewer check the claims one by one. Each script runs on its own, prints human-readable results, and (optionally) writes a JSON report you can save.

The goal is to answer five practical questions:

1) **Is the pipeline non-circular?** (No hidden use of the measured fine-structure constant α in the steps that should be α-free.)
2) **Is there a predictive step?** (Given a Scale value that did not use α, do we get a definite prediction for α⁻¹?)
3) **Is the mapping consistent?** (If we invert the mapping, do we recover the original Scale?)
4) **Are alternatives clearly worse?** (If we try another construction, does it visibly miss?)
5) **Is the result stable?** (Do small changes in inputs produce appropriately small changes in outputs? Does α⁻¹ scale as expected with Scale?)

---

## What’s in this folder

- `scale_from_independent_inputs.py`  
  Locks a **Scale** value from a small JSON file and records its provenance (where it came from). This should be **α-free** for predictive audits.

- `alpha_from_scale.py`  
  Computes **α⁻¹** from a given Scale using the geometry/density relation.

- `scale_from_alpha.py`  
  Computes **Scale** from a given α⁻¹. Use this for **round-trip** checks, not for predictive audits.

- `alt_hypothesis_sweep.py`  
  Compares two constructions against a target α⁻¹:  
  1) an “inverse-area” construction (the one used for prediction), and  
  2) a “Core×Scale” construction (a deliberate alternative).

- `static_guard_all.py`  
  Scans the included core modules and fails if any banned identifier appears at the **code** level (e.g., `alpha`, `alpha_inv`, `fine_structure`). This is a **non-circularity** attest.

- `uncertainty_budget.py`  
  Propagates declared uncertainties in inputs (Scale, φ, QC₂) to a **Δ(α⁻¹)** with printed partial derivatives.

- `robustness_sweep.py`  
  Sweeps Scale in a small window and fits a log–log slope. Expect a slope near **2** because α⁻¹ ∝ Scale².

Also included: minimal core modules these scripts import (`density_certificate_v2.py`, `g_electron.py`, `electron_density_certificate.py`, `lst_sigma_em0.py`) and a sample data file `independent_scale.sample.json`.

---

## Terms used (brief)

- **Scale**: a single number used by the geometry side of the calculation. For predictive audits, it must be obtained **without** using the measured α anywhere.
- **α⁻¹**: inverse fine-structure constant (e.g., 137.035999084).  
- **Provenance**: a short note describing where a number came from so others can reproduce it.

---

## Before you begin

- Python 3.8+ recommended.
- No extra packages required; everything uses the standard library.
- Run commands **inside** this folder.

---

## Case A — Predictive audit (preferred)

**Purpose:** Show a real prediction for α⁻¹ from a Scale value that did not use α.

1) Prepare a Scale JSON that **did not** use α to produce its value. Example:
   ```json
   {
     "scale": "240.463",
     "provenance": "Geometry-side inference only (α-free). Derived with density_certificate_v2, commit <hash>, inputs <…>, 2025-10-26."
   }
   ```
   Save as `my_scale.json` (or use the included sample for a demo).

2) Record and attest the input:
   ```bash
   python scale_from_independent_inputs.py --from-json my_scale.json --json scale_manifest.json
   ```
   This prints the value and a SHA256 of your file.

3) Predict α⁻¹ from that Scale:
   ```bash
   python alpha_from_scale.py --scale 240.463 --json alpha_from_scale.json
   ```
   **What to expect:** a definite number. With `scale=240.463` you should see:
   ```
   alpha^-1 = 136.760142...
   ```
   **What this tells us:** there is a clear, deterministic prediction from α-free inputs. No fitting to α occurred here.

---

## Case B — Round-trip closure (sanity check)

**Purpose:** Show the mapping is internally consistent.

1) Compute the Scale that would reproduce CODATA α⁻¹:
   ```bash
   python scale_from_alpha.py --alpha-inv 137.035999084
   # → Scale ≈ 240.7053948116...
   ```

2) Plug that Scale back in:
   ```bash
   python alpha_from_scale.py --scale 240.7053948116247830977963082577576792836620860533
   ```
   **What to expect:** α⁻¹ ≈ 137.035999084 to numerical precision.

**What this tells us:** the formulas invert each other correctly. This is a **sanity** check. It is not predictive, because we started from α.

---

## Case C — Alternative hypothesis comparison

**Purpose:** Show that the chosen construction outperforms a clear alternative.

Run:
```bash
python alt_hypothesis_sweep.py --alpha-inv-target 137.035999084 --scale 240.463 --json alt.json
```

**What to expect (typical):**
- **Inverse-area:** α⁻¹ close to the target (for `scale=240.463`, ~136.760142…, about 0.201% low).
- **Core×Scale:** α⁻¹ off by many orders of magnitude (a miss near 10⁻⁶), showing it is not competitive.

**What this tells us:** among simple constructions, the inverse-area choice is strongly preferred by data; the alternative is visibly ruled out.

---

## Case D — Non-circularity guard

**Purpose:** Ensure the α-free pathway is actually α-free in code.

Run:
```bash
python static_guard_all.py --paths lst_sigma_em0.py g_electron.py electron_density_certificate.py density_certificate_v2.py --json guard.json
```

**What to expect:** no **AST-level** hits for banned identifiers. Token hits in comments or docstrings are allowed; AST hits would fail the run.

**What this tells us:** the α-free modules do not reference α at the code level.

---

## Case E — Uncertainty budget

**Purpose:** Quantify how small input changes move α⁻¹.

Example:
```bash
python uncertainty_budget.py --scale 240.463 --dscale 1e-3 --json ub.json
```

**What to expect:**  
- A printed `alpha_inv` for the nominal Scale.  
- Partial derivatives like `da_dscale`, and absolute/relative uncertainty (e.g., ~1.14×10⁻³ absolute for the example above).

**How to read it:** If you know the uncertainty in your Scale, you can immediately read off the implied uncertainty in α⁻¹. Because α⁻¹ ∝ Scale², a relative error ε in Scale produces about **2ε** in α⁻¹.

---

## Case F — Robustness sweep

**Purpose:** Check the expected scaling and basic numerical stability.

Run:
```bash
python robustness_sweep.py --scale 240.463 --rel-span 0.05 --points 9 --json sweep.json
```

**What to expect:**  
- A log–log slope very close to **2.0**.  
- A smooth table of `(Scale, alpha_inv)` pairs in the JSON.

**What this tells us:** the implementation behaves as expected (α⁻¹ ∝ Scale²), and small changes in Scale do not cause erratic jumps.

---

## Interpreting the numbers you may see

- With the sample `scale = 240.463`, you should see **α⁻¹ ≈ 136.760142…**.  
- If you invert with **137.035999084**, you should get **Scale ≈ 240.7053948116…**.  
- The difference between those two Scales is about **0.10%**, which implies roughly **0.20%** difference in α⁻¹. That is what you see in practice.

These small mismatches are **expected** when the Scale you feed in was obtained independently (and α-free). They demonstrate **predictivity**, not tuning.

---

## Frequently asked questions

**Q: What does “α-free” mean in practice?**  
A: The process used to produce your Scale did not reference the measured α value anywhere (no fitting, no seeding with α, no hidden dependency). Document that process in the `provenance` text.

**Q: Where do I put my Scale?**  
A: Create your own JSON, e.g. `my_scale.json`:
```json
{
  "scale": "240.463",
  "provenance": "Describe how you produced this number without using α."
}
```
Then run:
```bash
python scale_from_independent_inputs.py --from-json my_scale.json --json scale_manifest.json
python alpha_from_scale.py --scale 240.463 --json alpha_from_scale.json
```

**Q: My numbers differ by a tiny amount from the examples. Is that a problem?**  
A: No. Different platforms and precisions can lead to very small differences. Large, order-of-magnitude differences where you expect a close match would be a concern.

**Q: What proves this is not circular?**  
A: Two things: (1) your **provenance** statement for Scale, and (2) the **static guard** confirming that the core α-free modules contain no code-level references to α.

**Q: What if I want to demonstrate perfect agreement with CODATA α?**  
A: Use `scale_from_alpha.py` to compute the matching Scale, then feed that Scale into `alpha_from_scale.py`. This is a **closure test**, not a predictive audit.

---

## Troubleshooting

- **ModuleNotFoundError** when running a script  
  Ensure you are running **inside** this folder so Python can import the included minimal modules.

- **Static guard failure** (AST hits found)  
  Read the `guard.json`. If the hit is in a docstring or comment, it will be marked as a token, not an AST name/attribute. AST hits indicate a real code reference that needs removal.

- **Unexpected large errors in `alt_hypothesis_sweep.py`**  
  Verify the `--scale` value and the `--alpha-inv-target` you intended to test. Typos of one or two digits can cause very large relative errors.

---

## Appendix — Relations used

All computations here use exact Decimal arithmetic and the following simple relations:

- Golden ratio: φ = (1 + √5)/2  
- Half power: (1/2)·φ⁶  
- Geometry relation:  
  \[
  \alpha^{-1} \;=\; \frac{\text{Scale}^2}{15\,\pi\,\big(\tfrac12\,\phi^6\big)}\,.
  \]
- Inverse relation:  
  \[
  \text{Scale} \;=\; \sqrt{\alpha^{-1}\cdot 15\,\pi\,\big(\tfrac12\,\phi^6\big)}\,.
  \]
- Expected scaling: α⁻¹ ∝ Scale² ⇒ a relative change ε in Scale gives ≈ 2ε in α⁻¹.

---

### What each case ultimately tells us (one sentence each)

- **Predictive audit:** The pipeline produces a definite α⁻¹ from α-free inputs.  
- **Round-trip closure:** The mapping is internally consistent.  
- **Alternative comparison:** The chosen construction wins decisively over a clear competitor.  
- **Non-circularity guard:** No hidden α usage contaminates the α-free path.  
- **Uncertainty budget:** Small input uncertainties translate to appropriately sized output uncertainties.  
- **Robustness sweep:** The expected α⁻¹ ∝ Scale² scaling holds numerically.
