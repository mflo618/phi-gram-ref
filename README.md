# Repo layout

```
/1 - LST_GRC_Phi_Gram/
    README.md
    (phi_gram_ref.py, mapping_demo.py, etc.)

/2 - LST_FU_Alpha_FSC/
    README_AUDIT.md
    HOW_TO_VERIFY.md
    RUN_AUDIT_SUITE.sh
    MANIFEST.json
    CHECKSUMS.txt
    (A-Pack (LST_Alpha_A_ProofPack_v1_4_0) → researcher/dev pack for core proofs + derivations.
    (B-Pack (LST_Alpha_B_Audit_Addons_v1_0) → auditor pack for a quick, self-contained, traceable audit (predictivity, non-circularity, robustness))

/3 - LST_FU_ProofPack_A_v3/
    README.md
    run_doc_style.sh
    (evolve.py, mass_engine.py, toggles.yaml, …)

/README_START_HERE.md
```

---

# README_START_HERE.md

**Two ways to verify, depending on your focus:**

**A) Theory-first (matches paper order):**
1 → 2 → 3

* **1 – GRC / Phi-Gram:** Reproduce the geometric/dynamic selector. Expect SU(2)/(4,5) Pass; SU(3)/(5,6) Pass; SU(4)/(6,7) “static deep but **dynamic fail**.”
* **2 – Alpha/FSC (traceable):** Make an α-free prediction from a provided Scale; confirm non-circularity and scaling; do a round-trip closure with CODATA.
* **3 – FU ProofPack v3:** Run β/bridge (uses convergence-rates from #1) and Mass→CKM checks.

**B) Quick trust bootstrap (10 minutes):**
2 only, then 1 → 3 as desired.

---

# 1 – LST_GRC_Phi_Gram/README.md (at a glance + run)

**What this proves:** A geometric resonance filter **selects SU(3) over SU(4)**: SU(2)/(4,5) and SU(3)/(5,6) show viable, dynamically stable resonance; SU(4)/(6,7) finds a very deep static alignment but **fails** the dynamic stability threshold.

**How to run (examples):**

```bash
# SU(2): harmonics (4,5) → Pass
python phi_gram_ref.py --harmonics 4,5 --group su3 --cesaro-start-T --check-dynamics

# SU(3): harmonics (5,6) → Pass
python phi_gram_ref.py --harmonics 5,6 --group su3 --cesaro-start-T --check-dynamics

# SU(4): harmonics (6,7) → dynamic Fail
python phi_gram_ref.py --harmonics 6,7 --group su3 --cesaro-start-T --check-dynamics
```

**Expected signatures (representative):**

* SU(2) (4,5): rel_min ~ **1.09e-02**, convergence_rate ~ **8.69e-11** → **Coherent (Pass)**
* SU(3) (5,6): rel_min ~ **2.09e-02**, convergence_rate ~ **1.56e-10** → **Coherent (Pass)**
* SU(4) (6,7): rel_min ~ **7.08e-05** (very small / “static deep”), **but** convergence_rate ~ **4.75e-10** (fails threshold 3e-10) → **Unstable (Fail)**

**Why this matters:** It separates “exists a coherent state” from “dynamically accessible,” matching the observed SU(3)–SU(2) and rejecting SU(4).

**Next:** go to **2 – LST_FU_Alpha_FSC** for a crisp α-free numeric prediction and guards.

---

# 2 – LST_FU_Alpha_FSC/README_AUDIT.md

**At a glance (expected outputs):**

* `python alpha_from_scale.py --scale 240.463` → **α⁻¹ ≈ 136.760142…**
* `python scale_from_alpha.py --alpha-inv 137.035999084` → **Scale ≈ 240.7053948116…** (closure)
* `python robustness_sweep.py …` → log–log slope **≈ 2.0**
* `python static_guard_all.py …` → **no AST-level hits** (token hits in comments are informational)

**What this proves:** Predictivity from α-free inputs, non-circularity, stability of the mapping, and traceability (manifests + checksums).

**Next:** go to **3 – LST_FU_ProofPack_A_v3** for β/bridge and Mass→CKM.

---

# 3 – LST_FU_ProofPack_A_v3/README.md

**What this proves:**

* **β / Bridge (evolve.py):** Uses convergence-rates from GRC to form a one-loop dynamic discriminant; parameter-free SU(4) prediction shows the expected separation.
* **Mass → CKM (mass_engine.py):** One-run flavor checkpoint with band checks (e.g., J within documented range when toggles are on).

**How to run (doc-style):**

```bash
# Bridge / beta runner (example)
python evolve.py --inputs inputs/alphas_MZ.json --scheme inputs/scheme.json --targets targets

# Flavor runner
python mass_engine.py --config toggles.yaml

# Or batch
bash run_doc_style.sh
```



---

# LST Computational Tools

This repository contains computational tools and reference implementations for the Light-Space Theory (LST) framework.

## About

These scripts are the official computational certificates for the foundational papers on LST. They are provided to ensure full transparency and allow for independent verification of the theory's numerical and algebraic results.

For the complete theoretical context and the full collection of papers, please see the Light-Space Theory community on Zenodo.

## Usage

Each script is designed to be run from the command line. For specific instructions, arguments, and the physical context for each tool, please refer to the corresponding scientific paper.

**Prerequisites:**
- Python 3
- NumPy (`pip install numpy`)

**Generic Example:**
```bash
python [script_name].py --[arguments]
```

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

The core principle of this license is "ShareAlike" (or "copyleft"). You are free to run, study, share, and modify this software. If you distribute a modified version, you must also share your modifications under the same GPLv3 license.

This ensures that the project and its derivatives will always remain open-source and accessible to the entire community. **The full text of the license is available in the `LICENSE` file.**

## Contact

For more information on the theoretical framework, please visit [mflo.life](https://mflo.life).
```


