# Critic Guide — How to Audit This Pack

1) **Non-circular K**: Run `python run_all.py --show-static-scan`. Verify the AST scan shows no identifiers `alpha` or `g`
   inside `compute_K_electron()`.
2) **One-loop constant**: Check integral ≈ 1/30, `K_numeric == 1/(60π²)` within machine precision.
3) **Geometry assembly**: `(1/2)φ⁶ = 4φ + 5/2` identity check; no couplings used.
4) **Round-trip identity**: `G_target` from the input α⁻¹ reproduces α⁻¹ to ~1e−12 when closed.
5) **Bridge tests**: Compare `G_original` vs `G_target` (should fail by ~10^8), and `G_inverse_area` vs `G_target`
   (should be within ~0.2% using Scale≈240.463).
6) **Tamper tests**: perturb G by ±1% (and ±ppm) and watch α⁻¹ shift accordingly.


---

## JSON proof log schema (v1.4.0)
The runner writes a JSON object with keys:
- `runner_version`: string
- `timestamp_utc`: ISO8601 string
- `alpha_inv_input`: string (Decimal)
- `sections`: object with optional section keys:
    - `interaction`: { I_midpoint, K_numeric, K_closed, rel_diff, K_with_flavors?, rel_factor_vs_e_only? }
    - `geometry`: { phi, half_phi6 }
    - `closure_target`: { alpha_inv, G_target, required_QC2_times_rho, scale_solved, scale_choice, scale_used }
    - `round_trip`: { alpha_back, abs_err, rel_err, status }
    - `static_scan`: { flags, body_excerpt } or { error }
    - `tamper`: [ { pct, alpha_inv, delta }, ... ]
    - `bridge`: { G_target, G_original, G_inverse_area, rel_err_orig, rel_err_invArea }
    - `density_cert_v1`: { scale, G_cert, alpha_inv, rel_err_vs_target }
    - `density_cert_v2`: { scale, G_cert, alpha_inv, rel_err_vs_target }
    - `compute_scale`: { scale_solved, rel_err_vs_240_463 }
