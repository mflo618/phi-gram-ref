
# SU(N) → Harmonic Mapping — Axiom Demo (v1.0)

This mini-pack isolates the **group–harmonic mapping** argument from the rest of the ProofPack.

## What it shows

- Declared **a priori axioms** (A1 adjacency; A2 reserved kinematic modes; A3 monotone).
- Runtime **pass/fail** checks for alternative proposals (N-fold, rank, dimension, Coxeter).
- Derived **affine mapping** `f(N)=N+c` with minimal `c` satisfying the axioms (typically `c=2` under default A1/A2).

## Quick start

```bash
python3 mapping_demo.py --show # no show flag needed; runs demo by default
python3 mapping_demo.py --json examples/mapping_log.json
```

## Axiom toggles

```bash
# Default: A1 unit step; A2={1,2,3}; A3 monotone
python3 mapping_demo.py

# Relax adjacency (A1 OFF), keep reserved
python3 mapping_demo.py --mapping-relax-adjacency

# Remove reserved set entirely (A2 OFF) + relax A1 (dimension passes)
python3 mapping_demo.py --mapping-relax-adjacency --mapping-reserved ""
```

Outputs are colorized; use `--no-color` if your environment strips ANSI.
