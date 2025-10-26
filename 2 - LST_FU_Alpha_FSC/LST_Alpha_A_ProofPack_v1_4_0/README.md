# Alpha Non-Circular Proof Pack (v1.3.2)

This package demonstrates the **non-circular derivation** of the fine-structure constant closure and now includes a
**Computational Certificate for the Electron Density Ratio** that justifies the successful bridge used in Section [7].

> G (geometry) = I (interaction) = g²·K  ⇒  α⁻¹ = 1/(15·π·G)  
> K = 1/(60π²) (α-free),  G = (1/2)·QC₂·(ρ/ρ_min)·φ⁶ (α-free)

New in **v1.3.0**:
- Adds `electron_density_certificate.py` with first-principles justification:
  - **QC₂(e) = 1** as the *topological count* of a single U(1) one-cycle defect (do **not** double-count the Core factor from the mass formula).
  - **ρ/ρ_min = 1/Scale²** (U(1) Thomson *area law*): density is *count per area*, the dilation Scale only rescales area → density scales as Scale⁻².
- Updates `run_all.py` with a `--density-cert` section to print the certificate result and compare to the target.
- Keeps `--show-bridge` to contrast the failed “Core×Scale with QC₂=2φ” vs. the successful inverse-area rule.

Quick run:
```bash
python run_all.py --show-static-scan --show-tamper --show-bridge --density-cert
```


**v1.3.1** adds `density_certificate_v2.py` and a new `--density-cert-v2` flag with first-principles justification.

**v1.4.0** adds a hardened CLI and JSON logging.

Run the full demo:
```bash
python run_all.py --show-static-scan --show-tamper --show-bridge   --density-cert --density-cert-v2 --compute-scale   --json prooflog.json
```

Options:
- `--no-color` to disable ANSI colors.
- `--version` prints runner version.
