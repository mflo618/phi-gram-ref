#!/usr/bin/env bash
set -e
echo "## gauge_filter_cert — baseline_intact"
( cd gauge_filter_cert && python cert.py --group SU3 --toggles toggles.yaml || true )
( cd gauge_filter_cert && python cert.py --group SU4 --toggles toggles.yaml || true )
( cd gauge_filter_cert && pytest -q || true )

echo "## beta_h_runner — baseline_intact"
( cd beta_h_runner && python evolve.py --inputs inputs/alphas_MZ.json --scheme inputs/scheme.json --targets targets )

echo "## ckm_from_mass — baseline_intact"
( cd ckm_from_mass && python mass_engine.py --config toggles.yaml || true )
