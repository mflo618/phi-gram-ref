# lst_sigma_em0.py (v1.0.2)
# Reference implementation for the Σ_EM(0) Certificate
# Non-circular interaction side for G = I

from decimal import Decimal, getcontext

# High precision for reproducible digits
getcontext().prec = 60

# Use a high-precision Decimal value of π to avoid float->Decimal roundoff
DECIMAL_PI = Decimal("3.14159265358979323846264338327950288419716939937510")

def compute_K_electron():
    """
    Return K for the electron-only loop at the Thomson limit:
        K = 1 / (60 * pi^2)
    By construction, this K contains no coupling dependence (no alpha, no g).
    """
    return Decimal(1) / (Decimal(60) * (DECIMAL_PI ** 2))

def compute_K_with_flavors(masses_mev):
    """
    Return K including additional charged flavors f via:
        K = (1 / (60*pi^2)) * sum_f (m_e^2 / m_f^2)
    masses_mev: dict with keys 'e','mu','tau',... and positive masses in MeV.
    The electron mass must be present with key 'e'.
    """
    me = Decimal(masses_mev['e'])
    factor = Decimal(0)
    for k, mf in masses_mev.items():
        mf = Decimal(mf)
        factor += (me/me)**2 if k == 'e' else (me/mf)**2
    return compute_K_electron() * factor

def solve_alpha_inverse(G):
    """
    Given G (the geometric capacity for the electron, α-free),
    solve for the fine-structure constant via:
        alpha^{-1} = 1 / (15 * pi * G).
    """
    G = Decimal(G)
    return Decimal(1) / (Decimal(15) * DECIMAL_PI * G)

def target_G(alpha_inv):
    """
    Given a target alpha^{-1}, return the implied G that would reproduce it:
        G = 1 / (15 * pi * alpha^{-1}).
    Useful for cross-checks and sanity tests.
    """
    a_inv = Decimal(alpha_inv)
    return Decimal(1) / (Decimal(15) * DECIMAL_PI * a_inv)
