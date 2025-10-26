#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Traceability: real deterministic computation (no mocks), no network I/O.
# Deterministic Decimal arithmetic. Non-circular by construction (no alpha usage).
# See HOW_TO_VERIFY.md, MANIFEST.json, CHECKSUMS.txt.
# -----------------------------------------------------------------------------
from decimal import Decimal, getcontext
getcontext().prec = 80
import argparse, json
from density_certificate_v2 import alpha_inv_from_scale
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--scale', required=True)
    ap.add_argument('--json')
    a=ap.parse_args()
    val = alpha_inv_from_scale(Decimal(a.scale))
    print(f'alpha^-1 = {val}')
    if a.json:
        with open(a.json,'w') as f: json.dump({'scale':a.scale,'alpha_inv':str(val)}, f, indent=2)
