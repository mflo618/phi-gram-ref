#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Traceability: real deterministic computation (no mocks), no network I/O.
# Deterministic Decimal arithmetic. Non-circular by construction (no alpha usage).
# See HOW_TO_VERIFY.md, MANIFEST.json, CHECKSUMS.txt.
# -----------------------------------------------------------------------------
from decimal import Decimal, getcontext
getcontext().prec = 80
DECIMAL_PI = Decimal('3.14159265358979323846264338327950288419716939937510')
def phi_dec():
    return (Decimal(1)+Decimal(5).sqrt())/Decimal(2)
def half_phi6():
    phi=phi_dec(); return (phi**6)/Decimal(2)
def alpha_inv_from_scale(scale):
    denom = Decimal(15)*DECIMAL_PI*half_phi6()
    s = Decimal(scale)
    return (s*s)/denom
