#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Traceability: real deterministic computation (no mocks), no network I/O.
# Deterministic Decimal arithmetic. Non-circular by construction (no alpha usage).
# See HOW_TO_VERIFY.md, MANIFEST.json, CHECKSUMS.txt.
# -----------------------------------------------------------------------------
from decimal import Decimal, getcontext
getcontext().prec = 80
from density_certificate_v2 import phi_dec, half_phi6
__all__=['phi_dec','half_phi6']
