# How to Verify (Traceability)
- These scripts are real computations (not mocks). No network calls. Deterministic Decimal arithmetic.
- To attest non-circularity and reproduce results, run:
  1) python static_guard_all.py --paths lst_sigma_em0.py g_electron.py electron_density_certificate.py density_certificate_v2.py --json guard.json
  2) python scale_from_independent_inputs.py --from-json independent_scale.sample.json --json scale_manifest.json
  3) python alpha_from_scale.py --scale 240.463 --json alpha_from_scale.json
  4) python scale_from_alpha.py --alpha-inv 137.035999084 --json scale_from_alpha.json
  5) python alt_hypothesis_sweep.py --alpha-inv-target 137.035999084 --scale 240.463 --json alt.json
  6) python robustness_sweep.py --scale 240.463 --rel-span 0.05 --points 9 --json sweep.json
  7) python uncertainty_budget.py --scale 240.463 --dscale 1e-3 --json ub.json
- Check CHECKSUMS.txt and MANIFEST.json for file hashes/sizes.
