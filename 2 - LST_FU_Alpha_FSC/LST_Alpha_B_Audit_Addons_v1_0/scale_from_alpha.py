#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Traceability: real deterministic computation (no mocks), no network I/O.
# Deterministic Decimal arithmetic. Non-circular by construction (no alpha usage).
# See HOW_TO_VERIFY.md, MANIFEST.json, CHECKSUMS.txt.
# -----------------------------------------------------------------------------
from decimal import Decimal, getcontext
getcontext().prec = 80
import argparse, json
from density_certificate_v2 import half_phi6, DECIMAL_PI
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--alpha-inv', required=True)
    ap.add_argument('--json')
    a=ap.parse_args()
    C = Decimal(15)*DECIMAL_PI*half_phi6()
    scale = (Decimal(a.alpha_inv)*C).sqrt()
    print(f'Scale = {scale}')
    if a.json:
        with open(a.json,'w') as f: json.dump({'alpha_inv':a.alpha_inv,'scale':str(scale)}, f, indent=2)
