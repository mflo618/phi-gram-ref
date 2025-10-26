# density_certificate_v2.py
from decimal import Decimal, getcontext
getcontext().prec = 80

DECIMAL_PI  = Decimal("3.14159265358979323846264338327950288419716939937510")
DECIMAL_PHI = Decimal("1.61803398874989484820458683436563811772030917980576")

def phi_dec(): return DECIMAL_PHI
def half_phi6(): return (DECIMAL_PHI ** 6) / Decimal(2)

def qc2_electron():
    """QC2(e) is a *count* of U(1) one-cycle defects for a singly charged electron → 1."""
    return Decimal(1)

def density_ratio(scale):
    """Area-law: rho/rho_min = 1/scale^2 (scale is the homothety factor of the sheet metric)."""
    s = Decimal(scale)
    return Decimal(1) / (s**2)

def G_from_scale(scale):
    """G = (1/2) * QC2 * (rho/rho_min) * phi^6 with QC2=1 and rho/rho_min=1/scale^2."""
    return half_phi6() * qc2_electron() * density_ratio(scale)

def alpha_inv_from_scale(scale):
    """alpha^{-1} = 1/(15*pi*G(scale))."""
    G = G_from_scale(scale)
    return Decimal(1) / (Decimal(15) * DECIMAL_PI * G)
