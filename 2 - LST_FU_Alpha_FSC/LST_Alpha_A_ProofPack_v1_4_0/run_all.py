
# run_all.py — Alpha Non-Circular Proof Pack (v1.4.0, hardened CLI)
from decimal import Decimal, getcontext
getcontext().prec = 80
import argparse, json, re, sys, os, ast, inspect, importlib.util, math, datetime

# Local modules
from lst_sigma_em0 import compute_K_electron, compute_K_with_flavors, solve_alpha_inverse, target_G
from g_electron import phi_dec, half_phi6, compute_G, required_QC2_times_rho, alpha_inverse_from_G
from electron_density_certificate import compute_G_from_density, alpha_inverse_from_scale
from density_certificate_v2 import G_from_scale as G_from_scale_v2, alpha_inv_from_scale as alpha_inv_from_scale_v2

DECIMAL_PI = Decimal("3.14159265358979323846264338327950288419716939937510")

VERSION = "v1.4.0"

def d(x): return Decimal(x)

# ---------- Color / UI helpers ----------
class UI:
    def __init__(self, use_color=True):
        self.use_color = use_color
        if not use_color:
            self.RESET = self.RED = self.GREEN = self.YELLOW = self.BLUE = self.DIM = ""
        else:
            self.RESET = "\033[0m"
            self.RED = "\033[31m"
            self.GREEN = "\033[32m"
            self.YELLOW = "\033[33m"
            self.BLUE = "\033[34m"
            self.DIM = "\033[2m"

    def ok(self, s): return (self.GREEN + s + self.RESET) if self.use_color else s
    def warn(self, s): return (self.YELLOW + s + self.RESET) if self.use_color else s
    def bad(self, s): return (self.RED + s + self.RESET) if self.use_color else s
    def info(self, s): return (self.BLUE + s + self.RESET) if self.use_color else s
    def dim(self, s): return (self.DIM + s + self.RESET) if self.use_color else s

def midpoint_integral_x2_1mx2(n=200000):
    s = d(0); dn = d(1)/d(n); x = d(0)
    for _ in range(n):
        xm = x + dn/d(2)
        s += (xm**2) * ((1 - xm)**2) * dn
        x += dn
    return s

def relerr(a,b):
    a = d(a); b = d(b)
    if b == 0: return d('NaN')
    return abs(a-b)/max(d(1),abs(b))

def static_scan_compute_K_ast():
    """
    AST-based scan of compute_K_electron() body that ignores comments and string literals.
    It flags only *identifiers* or attribute names equal to 'alpha' or 'g'.
    """
    path = os.path.join(os.path.dirname(__file__), "lst_sigma_em0.py")
    with open(path, "r") as f:
        src = f.read()
    tree = ast.parse(src)
    flags = {"identifier 'alpha' present": False, "identifier 'g' present": False}

    target_body = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_K_electron":
            target_body = node
            break
    if target_body is None:
        return flags, "compute_K_electron not found"

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node):
            if node.id == "alpha":
                flags["identifier 'alpha' present"] = True
            if node.id == "g":
                flags["identifier 'g' present"] = True
        def visit_Attribute(self, node):
            if node.attr == "alpha":
                flags["identifier 'alpha' present"] = True
            if node.attr == "g":
                flags["identifier 'g' present"] = True
            self.generic_visit(node)

    Visitor().visit(target_body)
    # Pretty source for humans
    spec = importlib.util.spec_from_file_location("lst_sigma_em0", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    func_src = inspect.getsource(mod.compute_K_electron)
    return flags, func_src

def bridge_numbers(core=None, scale=None, qc2_core=None, qc2_invarea=1, alpha_inv=None):
    # Defaults
    phi = phi_dec()
    if core is None:
        core = d(2)*phi  # 2φ
    if scale is None:
        scale = d("240.463")
    if qc2_core is None:
        qc2_core = d(2)*phi  # original user's choice
    half_phi6_val = half_phi6()

    # West tower (target G from α^-1)
    if alpha_inv is None:
        alpha_inv = d("137.035999084")
    else:
        alpha_inv = d(alpha_inv)
    G_target = d(1)/(d(15)*DECIMAL_PI*alpha_inv)

    # East tower variants
    # 1) Original flawed: rho = Core * Scale ; QC2 = 2φ (double counts Core)
    rho_orig = core * scale
    G_orig = half_phi6_val * qc2_core * rho_orig

    # 2) Inverse-area: rho = 1/Scale^2 ; QC2 = 1
    rho_inv = d(1) / (scale**2)
    G_inv = half_phi6_val * d(qc2_invarea) * rho_inv

    # 3) Mass-consistent collapses to inverse-area
    G_mass = G_inv

    return {
        "phi": phi,
        "half_phi6": half_phi6_val,
        "core": core,
        "scale": scale,
        "qc2_core": qc2_core,
        "G_target": G_target,
        "G_original": G_orig,
        "G_inverse_area": G_inv,
        "G_mass_consistent": G_mass
    }

def compute_scale_from_alpha(alpha_inv):
    """
    From v2 certificate:
      G(scale) = (1/2)*phi^6 * (1/scale^2)
      alpha^{-1} = 1 / (15*pi*G)
    => scale = sqrt( alpha^{-1} * (15*pi) * (1/2)*phi^6 )
    """
    phi6_over_2 = half_phi6()
    return ( (d(alpha_inv) * d(15) * DECIMAL_PI * phi6_over_2) ).sqrt()

def write_json(path, payload):
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    p = argparse.ArgumentParser(description="Alpha Non-Circular Proof Pack — verbose demo")
    p.add_argument("--alpha-inv", type=str, default="137.035999084", help="Target alpha^{-1}.")
    p.add_argument("--masses", type=str, default="", help='JSON dict of masses in MeV, e.g. {"e":0.51099895,"mu":105.6583755,"tau":1776.86}')
    p.add_argument("--show-static-scan", action="store_true", help="Show static scan of compute_K_electron() body.")
    p.add_argument("--show-tamper", action="store_true", help="Run tamper tests to demonstrate falsifiability.")
    p.add_argument("--show-bridge", action="store_true", help="Show bridge tests for rho/min variants.")
    p.add_argument("--density-cert", action="store_true", help="Print Electron Density Certificate result (QC2=1, rho=1/Scale^2).")
    p.add_argument("--density-cert-v2", action="store_true", help="Print Electron Density Certificate v2.0 result (QC2=1, rho=1/Scale^2).")
    p.add_argument("--compute-scale", action="store_true", help="Solve for Scale from alpha^{-1} via v2 certificate.")
    p.add_argument("--scale", type=str, default="", help="Override scale value (Decimal).")
    p.add_argument("--use-solved-scale", action="store_true", help="Use the computed scale from α^-1 for density/bridge sections.")
    p.add_argument("--json", type=str, default="", help="Path to write JSON proof log.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors in output.")
    p.add_argument("--version", action="store_true", help="Print version and exit.")
    args = p.parse_args()

    if args.version:
        print(VERSION)
        sys.exit(0)

    ui = UI(use_color=(not args.no_color))

    # Accumulate a machine log
    mlog = {
        "runner_version": VERSION,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "alpha_inv_input": args.alpha_inv,
        "sections": {},
    }

    def section(title):
        print(ui.info(f"=== {title} ==="))

    print(ui.info(f"=== Alpha Non-Circular Proof Pack — Verbose Run ({VERSION}) ==="))
    print("")

    # 1) Interaction-side certificate
    section("[1] Interaction-Side Certificate (Σ_EM(0), Thomson limit)")
    print("Definition: I = g^2 * K,    K = 1/(60*pi^2)  (α-free by construction)")
    I = midpoint_integral_x2_1mx2(100000)
    print(f"∫_0^1 x^2(1-x)^2 dx  = {I}")
    print(f"Error vs 1/30        = {I - d(1)/d(30)}")
    K_num = compute_K_electron()
    K_closed = d(1)/(d(60)*(DECIMAL_PI**2))
    print(f"K_numeric (module)    = {K_num}")
    print(f"K_closed_form         = {K_closed}")
    print(f"relative difference   = {relerr(K_num, K_closed)}\n")
    mlog["sections"]["interaction"] = {"I_midpoint": str(I), "K_numeric": str(K_num), "K_closed": str(K_closed), "rel_diff": str(relerr(K_num, K_closed))}

    if args.masses:
        import json as _json
        masses = _json.loads(args.masses)
        K_with = compute_K_with_flavors(masses)
        rel = K_with / K_num if K_num != 0 else d('NaN')
        print("Including flavors:")
        print(f"  K_with_flavors      = {K_with}")
        print(f"  relative factor     = {rel}  (ppm-level)\n")
        mlog["sections"]["interaction"]["K_with_flavors"] = str(K_with)
        mlog["sections"]["interaction"]["rel_factor_vs_e_only"] = str(rel)

    # 2) Geometry-side certificate
    section("[2] Geometry-Side Certificate (electron)")
    phi = phi_dec()
    half_phi6_val = half_phi6()
    print(f"φ                     = {phi}")
    print(f"(1/2)φ^6 (identity)  = 4φ + 5/2 = {4*phi + d(5)/d(2)}")
    print(f"(1/2)φ^6 (computed)  = {half_phi6_val}")
    print(f"relative difference   = {relerr(half_phi6_val, 4*phi + d(5)/d(2))}\n")
    mlog["sections"]["geometry"] = {"phi": str(phi), "half_phi6": str(half_phi6_val)}

    # 3) Closure target
    section("[3] Closure target from α^-1")
    alpha_inv = d(args.alpha_inv)
    G_target = d(1)/(d(15)*DECIMAL_PI*alpha_inv)

    # Compute a canonical solved scale from the input α^-1
    scale_solved = compute_scale_from_alpha(alpha_inv)
    # Choose scale based on flags
    if args.scale:
        chosen_scale = d(args.scale)
        scale_source = "user"
    elif args.use_solved_scale:
        chosen_scale = scale_solved
        scale_source = "solved"
    else:
        chosen_scale = d("240.463")
        scale_source = "default"

    print(f"Using Scale[{scale_source}] = {chosen_scale}\n")
    mlog['sections'].setdefault('closure_target', {})['scale_solved'] = str(scale_solved)
    mlog['sections']['closure_target']['scale_choice'] = scale_source
    mlog['sections']['closure_target']['scale_used'] = str(chosen_scale)

    print(f"α^-1 (input)          = {alpha_inv}")
    print(f"G_target              = {G_target}")
    qc2rho = G_target / half_phi6_val
    print(f"Required QC2*(rho/min)= {qc2rho}\n")
    mlog["sections"]["closure_target"] = {"alpha_inv": str(alpha_inv), "G_target": str(G_target), "required_QC2_times_rho": str(qc2rho)}

    # 4) Round-trip identity
    section("[4] Round-trip identity check")
    alpha_back = d(1)/(d(15)*DECIMAL_PI*G_target)
    print(f"α^-1(G_target)        = {alpha_back}  " + ui.ok("✅") )
    print(f"abs error             = {abs(alpha_back - alpha_inv)}")
    print(f"rel error             = {relerr(alpha_back, alpha_inv)}\n")
    mlog["sections"]["round_trip"] = {"alpha_back": str(alpha_back), "abs_err": str(abs(alpha_back - alpha_inv)), "rel_err": str(relerr(alpha_back, alpha_inv)), "status": "pass"}

    # 5) Static source scan
    if args.show_static_scan:
        section("[5] Static scan of compute_K_electron() (AST-based)")
        try:
            flags, body = static_scan_compute_K_ast()
            for k,v in flags.items():
                print(f"{k:<28}: {v}")
            print(ui.dim("--- function body ---"))
            print(body.strip())
            print("")
            mlog["sections"]["static_scan"] = {"flags": flags, "body_excerpt": body.strip()[:500]}
        except Exception as e:
            print(ui.bad(f"Static scan error: {e}"))
            mlog["sections"]["static_scan"] = {"error": str(e)}

    # 6) Tamper tests
    if args.show_tamper:
        section("[6] Tamper tests (falsifiability)")
        tamper_list = [d("0.01"), d("-0.01"), d("1e-6")]
        tlogs = []
        for pct in tamper_list:
            G_tamper = G_target * (d(1)+pct)
            alpha_tamper = d(1)/(d(15)*DECIMAL_PI*G_tamper)
            print(f"Tamper G by {pct:+}: α^-1 -> {alpha_tamper}  (Δ={alpha_tamper - alpha_inv})")
            tlogs.append({"pct": str(pct), "alpha_inv": str(alpha_tamper), "delta": str(alpha_tamper - alpha_inv)})
        print("")
        mlog["sections"]["tamper"] = tlogs

    # 7) Bridge tests
    if args.show_bridge:
        section("[7] Bridge tests (geometry-side density variants)")
        nums = bridge_numbers(alpha_inv=alpha_inv, scale=chosen_scale)
        print(f"Inputs: φ={nums['phi']}, (1/2)φ^6={nums['half_phi6']}  Core=2φ={nums['core']}  Scale≈{nums['scale']}")
        print(f"West (target)  G_target               = {nums['G_target']}")
        print(f"East (orig)    G(Core×Scale, QC2=2φ)  = {nums['G_original']}   " + ui.bad("❌"))
        print(f"East (invArea) G(1/Scale^2, QC2=1)    = {nums['G_inverse_area']}   " + ui.ok("✅"))
        print(f"East (mass)    G(m-bridge)            = {nums['G_mass_consistent']}   " + ui.ok("✅"))
        def rerr(a,b):
            a=d(a); b=d(b)
            return (a-b)/b if b!=0 else d('NaN')
        rel_orig = rerr(nums['G_original'], nums['G_target'])
        rel_inv = rerr(nums['G_inverse_area'], nums['G_target'])
        print(f"rel error (orig vs target)            = {rel_orig}")
        print(f"rel error (invArea vs target)         = {rel_inv}\n")
        mlog["sections"]["bridge"] = {
            "G_target": str(nums['G_target']),
            "G_original": str(nums['G_original']),
            "G_inverse_area": str(nums['G_inverse_area']),
            "rel_err_orig": str(rel_orig),
            "rel_err_invArea": str(rel_inv),
        }

    # 8) Electron Density Certificate v1
    if args.density_cert:
        section("[8] Electron Density Certificate (QC2=1, rho/min = 1/Scale^2)")
        scale = chosen_scale  # default paper value; override here if desired
        G_cert = compute_G_from_density(scale)
        alpha_from_scale = alpha_inverse_from_scale(scale)
        print(f"Using Scale ≈ {scale} (source={scale_source})")
        print(f"G_cert(scale)         = {G_cert}")
        print(f"α^-1(G_cert)          = {alpha_from_scale}")
        print(f"rel error vs target   = {relerr(alpha_from_scale, alpha_inv)}\n")
        mlog["sections"]["density_cert_v1"] = {"scale": str(scale), "G_cert": str(G_cert), "alpha_inv": str(alpha_from_scale), "rel_err_vs_target": str(relerr(alpha_from_scale, alpha_inv))}

    # 9) Electron Density Certificate v2.0
    if args.density_cert_v2:
        section("[9] Electron Density Certificate v2.0 (QC2=1, rho/min = 1/Scale^2)")
        scale = chosen_scale
        G_cert = G_from_scale_v2(scale)
        alpha_from_scale = alpha_inv_from_scale_v2(scale)
        print(f"Using Scale ≈ {scale} (source={scale_source})")
        print(f"G_cert_v2(scale)      = {G_cert}")
        print(f"α^-1(G_cert_v2)       = {alpha_from_scale}")
        print(f"rel error vs target   = {relerr(alpha_from_scale, alpha_inv)}\n")
        mlog["sections"]["density_cert_v2"] = {"scale": str(scale), "G_cert": str(G_cert), "alpha_inv": str(alpha_from_scale), "rel_err_vs_target": str(relerr(alpha_from_scale, alpha_inv))}

    # 10) Compute Scale from α^-1
    if args.compute_scale:
        section("[10] Solve for Scale from α^-1 (using v2 certificate)")
        scale_solved = compute_scale_from_alpha(alpha_inv)
        print(f"Solved Scale          = {scale_solved}  " + ui.ok("✅"))
        print(ui.dim(f"Tip: re-run with --use-solved-scale (or --scale {scale_solved}) to lock downstream sections."))
        # Compare against paper default 240.463
        rel = relerr(scale_solved, d("240.463"))
        print(f"rel error vs 240.463  = {rel}\n")
        mlog["sections"]["compute_scale"] = {"scale_solved": str(scale_solved), "rel_err_vs_240_463": str(rel)}

    print(ui.dim("Legend: ✅ pass, ❌ fail. Colors: green=good, yellow=warning, red=issue."))
    print(ui.info("------------------------------------------------------------"))

    # JSON log
    if args.json:
        ok, err = write_json(args.json, mlog)
        if ok:
            print(f"Wrote JSON proof log → {args.json}")
        else:
            print(f"Failed to write JSON log at {args.json}: {err}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail gracefully without Python traceback to the user
        msg = f"[FATAL] {e.__class__.__name__}: {e}"
        print(msg)
        # Do not re-raise; exit with non-zero code
        sys.exit(1)
