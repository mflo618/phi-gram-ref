# electron_density_certificate.py (v1.0.0)
# Computational Certificate for the Electron Density Ratio used in G = (1/2)·QC2·(ρ/ρ_min)·φ^6

from decimal import Decimal, getcontext
getcontext().prec = 80

DECIMAL_PI  = Decimal("3.14159265358979323846264338327950288419716939937510")
DECIMAL_PHI = Decimal("1.61803398874989484820458683436563811772030917980576")

def phi_dec():
    return DECIMAL_PHI

def half_phi6():
    return (DECIMAL_PHI ** 6) / Decimal(2)

def qc2_electron_topological_count():
    """
    QC2(e) is the *count* of U(1) one-cycle defects (singly charged electron) → 1.
    This is NOT the 'Core' factor from the mass formula; using Core here double-counts.
    """
    return Decimal(1)

def density_ratio_inverse_area(scale):
    """
    Electron action density at the Thomson limit is a *count per area* on the φ-phase sheet.
    A dilation by 'scale' rescales areas by scale^2. With the topological count fixed (= QC2), the density transforms as:
        rho / rho_min = 1 / scale^2
    """
    scale = Decimal(scale)
    return Decimal(1) / (scale ** 2)

def compute_G_from_density(scale):
    """
    Assemble G from the certificate:
        G = (1/2)·QC2·(rho/rho_min)·φ^6
          = (1/2)·1·(1/scale^2)·φ^6
    """
    return half_phi6() * qc2_electron_topological_count() * density_ratio_inverse_area(scale)

def alpha_inverse_from_scale(scale):
    """
    Using the closure α^{-1} = 1/(15·π·G) with the certified G(scale).
    """
    G = compute_G_from_density(scale)
    return Decimal(1) / (Decimal(15) * DECIMAL_PI * G)
