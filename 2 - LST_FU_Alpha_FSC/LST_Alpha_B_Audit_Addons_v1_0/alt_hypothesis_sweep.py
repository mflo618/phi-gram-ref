#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Traceability: real deterministic computation (no mocks), no network I/O.
# Deterministic Decimal arithmetic. Non-circular by construction (no alpha usage).
# See HOW_TO_VERIFY.md, MANIFEST.json, CHECKSUMS.txt.
# -----------------------------------------------------------------------------
from decimal import Decimal, getcontext
getcontext().prec = 80
import argparse, json
from density_certificate_v2 import phi_dec, half_phi6, DECIMAL_PI
def alpha_inv_from_G(G):
    return Decimal(1)/(Decimal(15)*DECIMAL_PI*G)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--alpha-inv-target', required=True)
    ap.add_argument('--scale', required=True)
    ap.add_argument('--core')
    ap.add_argument('--qc2-core')
    ap.add_argument('--qc2-inv', default='1')
    ap.add_argument('--json')
    a=ap.parse_args()
    d=Decimal; phi=phi_dec(); h=half_phi6()
    target = d(a.alpha_inv_target); scale=d(a.scale)
    core = d(a.core) if a.core is not None else d(2)*phi
    qc2_core = d(a.qc2_core) if a.qc2_core is not None else d(2)*phi
    qc2_inv = d(a.qc2_inv)
    rho_orig = core*scale; G_orig = h*qc2_core*rho_orig; a_orig = alpha_inv_from_G(G_orig)
    rho_inv = d(1)/(scale**2); G_inv = h*qc2_inv*rho_inv; a_inv = alpha_inv_from_G(G_inv)
    res={'target_alpha_inv':str(target),'scale':str(scale),'variants':{
        'orig_core_times_scale':{'G':str(G_orig),'alpha_inv':str(a_orig),'rel_error':str(abs((a_orig-target)/target))},
        'inverse_area':{'G':str(G_inv),'alpha_inv':str(a_inv),'rel_error':str(abs((a_inv-target)/target))},
    }}
    print(json.dumps(res, indent=2))
    if a.json:
        with open(a.json,'w') as f: json.dump(res, f, indent=2)
