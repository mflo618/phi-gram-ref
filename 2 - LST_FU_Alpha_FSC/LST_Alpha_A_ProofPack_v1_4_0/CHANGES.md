# Changelog

## v1.2.0
- Added Bridge Test logic and CLI (`--show-bridge` in run_all.py and `bridge_test.py`).
- Demonstrates original bridge failure and inverse-area success, with printed numeric comparisons.
- Bundled latest K and G certificates and AST static scan.


## v1.2.1
- Fix: `--show-bridge` now safely handles Decimal vs string for α⁻¹.


## v1.3.0
- Added Electron Density Certificate with QC₂(e)=1 and ρ/ρ_min=1/Scale² justification.
- Integrated certificate prints via `--density-cert` flag in run_all.py.


## v1.3.1
- Integrated Electron Density Certificate v2.0; added `--density-cert-v2` to verbose runner.


## v1.3.2
- Fix: robust handling of `--density-cert-v2` flag (present or absent) to prevent AttributeError.
## v1.4.0
- Hardened CLI: adds `--density-cert-v2`, `--compute-scale`, `--json <path>`, `--no-color`, and `--version`.
- Colorized, structured output with ✅/❌ markers; graceful fatal handling (no Python tracebacks).
- JSON proof log emitted when `--json` is provided (stable keys under `sections{}`).
- `compute-scale` solves Scale from α⁻¹ using the v2 certificate.
