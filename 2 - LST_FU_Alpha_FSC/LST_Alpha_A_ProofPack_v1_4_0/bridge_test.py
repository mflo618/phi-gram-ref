# bridge_test.py — Compare bridge variants for geometry-side density
from decimal import Decimal, getcontext
getcontext().prec = 80

from g_electron import phi_dec, half_phi6
from lst_sigma_em0 import solve_alpha_inverse, target_G

DECIMAL_PI = Decimal("3.14159265358979323846264338327950288419716939937510")
def d(x): return Decimal(x)

def bridge_numbers(core=None, scale=None, qc2_core=None, qc2_invarea=1, alpha_inv=None):
    phi = phi_dec()
    if core is None:
        core = d(2)*phi
    if scale is None:
        scale = d("240.463")
    if qc2_core is None:
        qc2_core = d(2)*phi
    half_phi6_val = half_phi6()
    if alpha_inv is None:
        alpha_inv = d("137.035999084")
    G_target = d(1)/(d(15)*DECIMAL_PI*alpha_inv)
    rho_orig = core*scale
    G_orig = half_phi6_val * qc2_core * rho_orig
    rho_inv = d(1)/(scale**2)
    G_inv = half_phi6_val * d(qc2_invarea) * rho_inv
    G_mass = G_inv
    return G_target, G_orig, G_inv, G_mass, phi, half_phi6_val, core, scale

def main():
    G_target, G_orig, G_inv, G_mass, phi, hphi6, core, scale = bridge_numbers()
    print("=== Bridge Test ===")
    print(f"phi={phi}")
    print(f"(1/2)phi^6={hphi6}")
    print(f"Core(2)=2phi={core}")
    print(f"Scale ≈ {scale}")
    print(f"G_target={G_target}")
    print(f"G_original (Core×Scale, QC2=2phi)={G_orig}")
    print(f"G_inverse_area (1/Scale^2, QC2=1)={G_inv}")
    print(f"G_mass_consistent={G_mass}")
    rel_orig = abs((G_orig - G_target)/G_target)
    rel_inv = abs((G_inv - G_target)/G_target)
    print(f"rel error (orig)={rel_orig}")
    print(f"rel error (invArea)={rel_inv}")
    # Assertions: original must fail by >1e6; inverse-area within 1%
    assert rel_orig > Decimal("1e6")
    assert rel_inv < Decimal("0.01")
    print("Bridge test assertions passed.")

if __name__ == "__main__":
    main()
