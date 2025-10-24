# LST Proof Pack — v3 (Parity, Hardened)

Commands remain identical to your doc.

Doc-style commands:
```bash
# 1) SU(3) vs SU(4)
cd gauge_filter_cert
python cert.py --group SU3 --toggles toggles.yaml
python cert.py --group SU4 --toggles toggles.yaml
pytest -q || true
cd ..

# 2) h-scaled β prereg test
cd beta_h_runner
python evolve.py --inputs inputs/alphas_MZ.json --scheme inputs/scheme.json --targets targets
pytest -q || true
cd ..

# 3) One-run Mass → CKM
cd ckm_from_mass
python mass_engine.py --config toggles.yaml
pytest -q || true
cd ..

# 4) Randomized mutation test
cd gauge_filter_cert
python randomized_mutation.py --trials 30            # default seed 314159 (reproducible)
# or
python randomized_mutation.py --trials 30 --seed 7   # different draw
```
