#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Traceability: real deterministic computation (no mocks), no network I/O.
# Deterministic Decimal arithmetic. Non-circular by construction (no alpha usage).
# See HOW_TO_VERIFY.md, MANIFEST.json, CHECKSUMS.txt.
# -----------------------------------------------------------------------------
from decimal import Decimal, getcontext
getcontext().prec = 80
import argparse, json
from density_certificate_v2 import half_phi6, phi_dec, DECIMAL_PI
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--scale', required=True)
    ap.add_argument('--dscale', default='0')
    ap.add_argument('--dphi', default='0')
    ap.add_argument('--dqc2', default='0')
    ap.add_argument('--rel-scale')
    ap.add_argument('--json')
    a=ap.parse_args()
    scale = Decimal(a.scale)
    dscale = scale*Decimal(a.rel_scale) if a.rel_scale is not None else Decimal(a.dscale)
    phi = phi_dec()
    phi6_over2 = half_phi6()
    denom = Decimal(15)*DECIMAL_PI*phi6_over2
    alpha_inv = scale*scale/denom
    da_dscale = (Decimal(2)*scale)/denom
    da_dphi = alpha_inv * (Decimal(-6)/phi)
    QC2 = Decimal(1); da_dqc2 = -alpha_inv/QC2
    da_abs = abs(da_dscale)*abs(dscale) + abs(da_dphi)*abs(Decimal(a.dphi)) + abs(da_dqc2)*abs(Decimal(a.dqc2))
    out={'inputs':{'scale':str(scale),'dscale':str(dscale),'dphi':a.dphi,'dqc2':a.dqc2},
         'alpha_inv':str(alpha_inv),
         'partials':{'da_dscale':str(da_dscale),'da_dphi':str(da_dphi),'da_dqc2':str(da_dqc2)},
         'uncertainty':{'abs':str(da_abs),'rel':str(da_abs/alpha_inv if alpha_inv!=0 else Decimal(0))}}
    print(json.dumps(out, indent=2))
    if a.json:
        with open(a.json,'w') as f: json.dump(out, f, indent=2)
