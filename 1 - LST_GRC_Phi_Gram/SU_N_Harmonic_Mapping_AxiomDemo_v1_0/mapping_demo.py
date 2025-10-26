
#!/usr/bin/env python3
# mapping_demo.py — SU(N) → Harmonic Mapping Axiom Demo (v1.0)
from decimal import Decimal, getcontext
getcontext().prec = 80
import argparse, json, sys, datetime

import group_mapping as gm

VERSION = "v1.0"

class UI:
    def __init__(self, use_color=True):
        self.use_color = use_color
        if not use_color:
            self.RESET = self.RED = self.GREEN = self.YELLOW = self.BLUE = self.DIM = ""
        else:
            self.RESET="\033[0m"; self.RED="\033[31m"; self.GREEN="\033[32m"; self.YELLOW="\033[33m"; self.BLUE="\033[34m"; self.DIM="\033[2m"
    def ok(self,s): return (self.GREEN+s+self.RESET) if self.use_color else s
    def warn(self,s): return (self.YELLOW+s+self.RESET) if self.use_color else s
    def bad(self,s): return (self.RED+s+self.RESET) if self.use_color else s
    def info(self,s): return (self.BLUE+s+self.RESET) if self.use_color else s
    def dim(self,s): return (self.DIM+s+self.RESET) if self.use_color else s

def write_json(path, payload):
    try:
        with open(path, "w") as f: json.dump(payload, f, indent=2, default=str)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    p = argparse.ArgumentParser(description="SU(N) → Harmonic Mapping Axiom Demo")
    p.add_argument("--mapping-reserved", type=str, default="1,2,3", help="Comma list for A2 (e.g., 1,2,3). Empty disables A2.")
    p.add_argument("--mapping-nmin", type=int, default=2, help="Minimum N (inclusive).")
    p.add_argument("--mapping-nmax", type=int, default=8, help="Maximum N (inclusive).")
    p.add_argument("--mapping-relax-adjacency", action="store_true", help="Relax A1: do not require unit step.")
    p.add_argument("--json", type=str, default="", help="Path to write JSON log of results.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    p.add_argument("--version", action="store_true", help="Print version and exit.")
    args = p.parse_args()

    if args.version:
        print(VERSION); sys.exit(0)

    ui = UI(use_color=(not args.no_color))
    print(ui.info(f"=== SU(N) → Harmonic Mapping Axiom Demo ({VERSION}) ===")); print("")

    # Parse Axioms
    reserved = tuple(int(x) for x in args.mapping_reserved.split(",") if x.strip()) if args.mapping_reserved.strip() else tuple()
    Nmin, Nmax = args.mapping_nmin, args.mapping_nmax
    require_step = 0 if args.mapping_relax_adjacency else 1
    print(f"Axioms: A1={'OFF' if require_step==0 else 'ON (unit step)'}; A2 reserved={reserved if reserved else '∅'}; A3=ON (monotone).")
    print("")

    # Evaluate proposals
    alt = {"fold N": gm.mapping_fold, "rank N-1": gm.mapping_rank, "coxeter N": gm.mapping_coxeter, "dim N^2-1": gm.mapping_dim}
    results = {}
    for name, fn in alt.items():
        ok, why = gm.passes_axioms(fn, reserved=reserved, Nmin=Nmin, Nmax=Nmax, require_step=require_step)
        status = ui.ok("✅") if ok else ui.bad("❌")
        print(f"{name:<12}  →  {status}  {why}")
        results[name] = {"ok": ok, "why": why}

    # Derive affine
    c, why = gm.pick_affine_c(reserved=reserved, Nmin=Nmin, Nmax=Nmax)
    f = lambda N: gm.mapping_affine(N,c)
    ok_aff, why_aff = gm.passes_axioms(f, reserved=reserved, Nmin=Nmin, Nmax=Nmax, require_step=require_step)
    print(f"\nAffine solution f(N)=N+c:  c={c}  " + (ui.ok("✅") if ok_aff else ui.bad("❌")) + f"  {why_aff}")
    rows = gm.mapping_table(f, Nmin=Nmin, Nmax=Nmax)
    print("N  f(N)  rank  dim  coxeter  pos_roots")
    for r in rows:
        print(f"{r['N']:>1}  {r['f(N)']:>4}  {r['rank']:>4}  {r['dim']:>3}  {r['coxeter']:>7}  {r['pos_roots']:>9}")
    if 3>=Nmin and 3<=Nmax:
        print(f"\nHighlight: SU(3) maps to {f(3)}, so the resonance pair is (f, f+1).")
    print("")
    print(ui.dim("Legend: ✅ pass, ❌ fail. Colors: green=good, yellow=warning, red=issue."))
    print(ui.info("------------------------------------------------------------"))

    if args.json:
        log = {"demo_version": VERSION,
               "timestamp_utc": datetime.datetime.utcnow().isoformat()+"Z",
               "axioms": {"A1_unit_step": bool(require_step), "A2_reserved": reserved, "A3_monotone": True, "Nmin": Nmin, "Nmax": Nmax},
               "alternatives": results, "affine_c": c, "table": rows}
        ok, err = write_json(args.json, log)
        if ok: print(f"Wrote JSON log → {args.json}")
        else: print(f"Failed to write JSON log: {err}")

if __name__ == "__main__":
    main()
