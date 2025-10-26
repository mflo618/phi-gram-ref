# g_electron.py (v1.0.1)
# Reference implementation for assembling the G(electron) scalar (geometry side).
# No α or g appears here.

from decimal import Decimal, getcontext
getcontext().prec = 80

# High-precision constants as Decimals (string literals to avoid float->Decimal issues)
DECIMAL_PI  = Decimal("3.14159265358979323846264338327950288419716939937510")
DECIMAL_PHI = Decimal("1.61803398874989484820458683436563811772030917980576")

def phi_dec():
    """Return golden ratio φ as Decimal (high precision)."""
    return DECIMAL_PHI

def half_phi6():
    """Return (1/2) * φ^6 as Decimal, computed from DECIMAL_PHI to ensure consistency."""
    phi = DECIMAL_PHI
    return (phi ** 6) / Decimal(2)

def compute_G(QC2, rho_ratio):
    """
    Assemble G for the electron:
        G = (1/2) * QC2 * (rho_ratio) * phi^6
    where rho_ratio = rho_e / rho_min.
    All inputs are expected to be positive Decimals (or coercible to Decimal).
    """
    QC2 = Decimal(QC2)
    rho_ratio = Decimal(rho_ratio)
    return half_phi6() * QC2 * rho_ratio

def target_G(alpha_inv):
    """
    Given a target alpha^{-1}, return G_target = 1 / (15 * pi * alpha^{-1}).
    """
    a_inv = Decimal(alpha_inv)
    return Decimal(1) / (Decimal(15) * DECIMAL_PI * a_inv)

def required_QC2_times_rho(alpha_inv):
    """
    Return the required product QC2 * (rho_e / rho_min) to meet a given alpha^{-1}:
        QC2*rho = G_target / ((1/2)*phi^6).
    """
    return target_G(alpha_inv) / half_phi6()

def alpha_inverse_from_G(G):
    """Local helper for round-trip checks: alpha^{-1} = 1 / (15 * pi * G)."""
    G = Decimal(G)
    return Decimal(1) / (Decimal(15) * DECIMAL_PI * G)
