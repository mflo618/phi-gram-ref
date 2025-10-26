# Audit Add-Ons — Read Me (Plain Text)

This folder contains small, single-purpose scripts that let a skeptical reviewer check the claims one by one. Each script runs on its own, prints human-readable results, and (optionally) writes a JSON report you can save.

The goal is to answer five practical questions:

1) Is the pipeline non-circular? (No hidden use of the measured fine-structure constant alpha in the steps that should be alpha-free.)
2) Is there a predictive step? (Given a Scale value that did not use alpha, do we get a definite prediction for alpha^-1?)
3) Is the mapping consistent? (If we invert the mapping, do we recover the original Scale?)
4) Are alternatives clearly worse? (If we try another construction, does it visibly miss?)
5) Is the result stable? (Do small changes in inputs produce appropriately small changes in outputs? Does alpha^-1 scale as expected with Scale?)

---

## What's in this folder

- `scale_from_independent_inputs.py`  
  Locks a Scale value from a small JSON file and records its provenance (where it came from). This should be alpha-free for predictive audits.

- `alpha_from_scale.py`  
  Computes alpha^-1 from a given Scale using the geometry/density relation.

- `scale_from_alpha.py`  
  Computes Scale from a given alpha^-1. Use this for round-trip checks, not for predictive audits.

- `alt_hypothesis_sweep.py`  
  Compares two constructions against a target alpha^-1:  
  1) an "inverse-area" construction (the one used for prediction), and  
  2) a "Core×Scale" construction (a deliberate alternative).

- `static_guard_all.py`  
  Scans the included core modules and fails if any banned identifier appears at the code level (e.g., alpha, alpha_inv, fine_structure). This is a non-circularity attest.

- `uncertainty_budget.py`  
  Propagates declared uncertainties in inputs (Scale, phi, QC2) to a delta(alpha^-1) with printed partial derivatives.

- `robustness_sweep.py`  
  Sweeps Scale in a small window and fits a log–log slope. Expect a slope near 2 because alpha^-1 is proportional to Scale^2.

Also included: minimal core modules these scripts import (`density_certificate_v2.py`, `g_electron.py`, `electron_density_certificate.py`, `lst_sigma_em0.py`) and a sample data file `independent_scale.sample.json`.

---

## Terms used (brief)

- Scale: a single number used by the geometry side of the calculation. For predictive audits, it must be obtained without using the measured alpha anywhere.
- alpha^-1: inverse fine-structure constant (for example, 137.035999084).
- Provenance: a short note describing where a number came from so others can reproduce it.

---

## Before you begin

- Python 3.8+ recommended.
- No extra packages required; everything uses the standard library.
- Run commands inside this folder.

---

## Case A — Predictive audit (preferred)

Purpose: Show a real prediction for alpha^-1 from a Scale value that did not use alpha.

1) Prepare a Scale JSON that did not use alpha to produce its value. Example:
   ```json
   {
     "scale": "240.463",
     "provenance": "Geometry-side inference only (alpha-free). Derived with density_certificate_v2, commit <hash>, inputs <...>, 2025-10-26."
   }
   ```
   Save as `my_scale.json` (or use the included sample for a demo).

2) Record and attest the input:
   ```bash
   python scale_from_independent_inputs.py --from-json my_scale.json --json scale_manifest.json
   ```
   This prints the value and a SHA256 of your file.

3) Predict alpha^-1 from that Scale:
   ```bash
   python alpha_from_scale.py --scale 240.463 --json alpha_from_scale.json
   ```
   What to expect: a definite number. With `scale=240.463` you should see something like
   ```
   alpha^-1 = 136.760142...
   ```
   What this tells us: there is a clear, deterministic prediction from alpha-free inputs. No fitting to alpha occurred here.

---

## Case B — Round-trip closure (sanity check)

Purpose: Show the mapping is internally consistent.

1) Compute the Scale that would reproduce CODATA alpha^-1:
   ```bash
   python scale_from_alpha.py --alpha-inv 137.035999084
   # -> Scale ~ 240.7053948116...
   ```

2) Plug that Scale back in:
   ```bash
   python alpha_from_scale.py --scale 240.7053948116247830977963082577576792836620860533
   ```
   What to expect: alpha^-1 ~ 137.035999084 to numerical precision.

What this tells us: the formulas invert each other correctly. This is a sanity check. It is not predictive, because we started from alpha.

---

## Case C — Alternative hypothesis comparison

Purpose: Show that the chosen construction outperforms a clear alternative.

Run:
```bash
python alt_hypothesis_sweep.py --alpha-inv-target 137.035999084 --scale 240.463 --json alt.json
```

What to expect (typical):
- Inverse-area: alpha^-1 close to the target (for `scale=240.463`, about 136.760142..., roughly 0.201% low).
- Core×Scale: alpha^-1 off by many orders of magnitude (a miss near 1e-6), showing it is not competitive.

What this tells us: among simple constructions, the inverse-area choice is strongly preferred by data; the alternative is visibly ruled out.

---

## Case D — Non-circularity guard

Purpose: Ensure the alpha-free pathway is actually alpha-free in code.

Run:
```bash
python static_guard_all.py --paths lst_sigma_em0.py g_electron.py electron_density_certificate.py density_certificate_v2.py --json guard.json
```

What to expect: no AST-level hits for banned identifiers. Token hits in comments or docstrings are allowed; AST hits would fail the run.

What this tells us: the alpha-free modules do not reference alpha at the code level.

---

## Case E — Uncertainty budget

Purpose: Quantify how small input changes move alpha^-1.

Example:
```bash
python uncertainty_budget.py --scale 240.463 --dscale 1e-3 --json ub.json
```

What to expect:
- A printed alpha_inv for the nominal Scale.
- Partial derivatives like da_dscale, and absolute/relative uncertainty (for example, about 1.14e-3 absolute for the example above).

How to read it: If you know the uncertainty in your Scale, you can immediately read off the implied uncertainty in alpha^-1. Because alpha^-1 is proportional to Scale^2, a relative error e in Scale produces about 2e in alpha^-1.

---

## Case F — Robustness sweep

Purpose: Check the expected scaling and basic numerical stability.

Run:
```bash
python robustness_sweep.py --scale 240.463 --rel-span 0.05 --points 9 --json sweep.json
```

What to expect:
- A log–log slope very close to 2.0.
- A smooth table of (Scale, alpha_inv) pairs in the JSON.

What this tells us: the implementation behaves as expected (alpha^-1 proportional to Scale^2), and small changes in Scale do not cause erratic jumps.

---

## Interpreting the numbers you may see

- With the sample scale = 240.463, you should see alpha^-1 ~ 136.760142...
- If you invert with 137.035999084, you should get Scale ~ 240.7053948116...
- The difference between those two Scales is about 0.10%, which implies roughly 0.20% difference in alpha^-1. That is what you see in practice.

These small mismatches are expected when the Scale you feed in was obtained independently (and alpha-free). They demonstrate predictivity, not tuning.

---

## Frequently asked questions

Q: What does "alpha-free" mean in practice?  
A: The process used to produce your Scale did not reference the measured alpha value anywhere (no fitting, no seeding with alpha, no hidden dependency). Document that process in the provenance text.

Q: Where do I put my Scale?  
A: Create your own JSON, for example `my_scale.json`:
```json
{
  "scale": "240.463",
  "provenance": "Describe how you produced this number without using alpha."
}
```
Then run:
```bash
python scale_from_independent_inputs.py --from-json my_scale.json --json scale_manifest.json
python alpha_from_scale.py --scale 240.463 --json alpha_from_scale.json
```

Q: My numbers differ by a tiny amount from the examples. Is that a problem?  
A: No. Different platforms and precisions can lead to very small differences. Large, order-of-magnitude differences where you expect a close match would be a concern.

Q: What proves this is not circular?  
A: Two things: (1) your provenance statement for Scale, and (2) the static guard confirming that the core alpha-free modules contain no code-level references to alpha.

Q: What if I want to demonstrate perfect agreement with CODATA alpha?  
A: Use `scale_from_alpha.py` to compute the matching Scale, then feed that Scale into `alpha_from_scale.py`. This is a closure test, not a predictive audit.

---

## Troubleshooting

- ModuleNotFoundError when running a script: ensure you are running inside this folder so Python can import the included minimal modules.

- Static guard failure (AST hits found): read the `guard.json`. If the hit is in a docstring or comment, it will be marked as a token, not an AST name/attribute. AST hits indicate a real code reference that needs removal.

- Unexpected large errors in `alt_hypothesis_sweep.py`: verify the `--scale` value and the `--alpha-inv-target` you intended to test. Typos of one or two digits can cause very large relative errors.

---

## Appendix — Relations used (plain text)

Constants and relations used here (plain text, no LaTeX):

- Golden ratio: phi = (1 + sqrt(5)) / 2
- Half power: (phi^6) / 2
- Geometry relation:
  alpha_inverse = Scale^2 / (15 * pi * (phi^6 / 2))
- Inverse relation:
  Scale = sqrt(alpha_inverse * 15 * pi * (phi^6 / 2))
- Expected scaling:
  alpha_inverse is proportional to Scale^2, so a relative change e in Scale gives approximately 2e in alpha_inverse.

---

What each case ultimately tells us (one sentence each):

- Predictive audit: The pipeline produces a definite alpha^-1 from alpha-free inputs.
- Round-trip closure: The mapping is internally consistent.
- Alternative comparison: The chosen construction wins decisively over a clear competitor.
- Non-circularity guard: No hidden alpha usage contaminates the alpha-free path.
- Uncertainty budget: Small input uncertainties translate to appropriately sized output uncertainties.
- Robustness sweep: The expected alpha^-1 proportional to Scale^2 scaling holds numerically.
