
The “Alpha / FSC” packs - **different audiences and purposes**:

## TL;DR

* **A-Pack (LST_Alpha_A_ProofPack_v1_4_0)** → **researcher/dev pack** for core proofs + derivations.
* **B-Pack (LST_Alpha_B_Audit_Addons_v1_0)** → **auditor pack** for a quick, self-contained, traceable audit (predictivity, non-circularity, robustness).

## What’s in each (practical differences)

| Aspect          | **A – ProofPack v1.4.0**                                                   | **B – Audit Addons v1.0**                                                                                                                                                         |
| --------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Primary goal    | Demonstrate/experiment with the core Alpha pipeline and related proofs     | Give skeptics a turnkey, α-free **prediction audit** with manifest + checks                                                                                                       |
| Audience        | Researcher / contributor                                                   | External auditor / reviewer                                                                                                                                                       |
| Entry scripts   | `run_all.py`, `run_example.sh`                                             | `RUN_AUDIT_SUITE.sh`                                                                                                                                                              |
| Key CLIs        | General proof runners (e.g., `bridge_test.py`), examples, derivation notes | **alpha_from_scale.py** (prediction), **scale_from_alpha.py** (closure), **robustness_sweep.py**, **uncertainty_budget.py**, **alt_hypothesis_sweep.py**, **static_guard_all.py** |
| Non-circularity | Not a dedicated guard                                                      | **Yes**: `static_guard_all.py` (AST-level checks)                                                                                                                                 |
| Traceability    | `MANIFEST_SHA256.txt`                                                      | **`MANIFEST.json` + `CHECKSUMS.txt`**, plus sample input & SHA-attestation                                                                                                        |
| Self-contained  | Often assumes repo context                                                 | **Yes**: unzip-and-run, no external deps                                                                                                                                          |
| Docs            | `README.md`, `CRITIC_GUIDE.md`, `DERIVATION_*.md`, `CHANGES.md`            | **README_AUDIT.md** (plain), **HOW_TO_VERIFY.md**                                                                                                                                 |

## When to use which

* **Send B-Pack first** to new/skeptical readers. It shows:

  * α-free input → **definite α⁻¹ prediction**,
  * non-circularity guard (**no AST α hits**),
  * robustness (slope ≈ 2) and uncertainty propagation,
  * alt-hypothesis comparison (inverse-area wins; Core×Scale fails).
* **Use A-Pack** when someone wants the **broader proof context**, derivations, or the bridge test workflow tied into the larger FU program.

## Are they redundant?

No. B-Pack **duplicates only the minimal core modules** needed to make audits self-contained. It adds guard/attest scripts and JSON outputs the A-Pack doesn’t provide. Keep both:

* B-Pack = **auditable artifact** (shareable, quick to run).
* A-Pack = **researcher toolkit** (richer docs/derivations and example flows).

-mFLO618
mflo.life
