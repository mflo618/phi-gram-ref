
from decimal import Decimal as D

def su_invariants(N:int):
    assert N>=2
    return {
        "N": N,
        "rank": N-1,
        "dim": N*N - 1,
        "coxeter": N,
        "pos_roots": N*(N-1)//2,
    }

def mapping_fold(N):      # "N-fold symmetry"
    return N

def mapping_rank(N):      # rank(SU(N)) = N-1
    return N-1

def mapping_dim(N):       # dim(SU(N)) = N^2 - 1
    return N*N - 1

def mapping_coxeter(N):   # h∨(SU(N)) = N
    return N

def mapping_affine(N, c): # general affine candidate
    return N + c

def passes_axioms(mapping_fn, *, Nmin=2, Nmax=8, reserved=(1,2,3), require_step=1):
    """
    Axioms A:
      A1 (adjacency):  if require_step==1, then f(N+1)-f(N) must equal 1
      A2 (reserved):   f(N) ∉ reserved for all N in [Nmin, Nmax]
      A3 (monotone):   f(N+1) > f(N) always
    """
    last = None
    for N in range(Nmin, Nmax+1):
        m = mapping_fn(N)
        if m in reserved:
            return False, f"violates A2 at N={N}: f(N)={m} in reserved {reserved}"
        if last is not None:
            if require_step==1 and m - last != 1:
                return False, f"violates A1 at N={N-1}→{N}: step {m-last} != 1"
            if m <= last:
                return False, f"violates A3 at N={N-1}→{N}: not strictly increasing"
        last = m
    return True, "passes Axioms A"

def pick_affine_c(*, Nmin=2, Nmax=8, reserved=(1,2,3)):
    """Find the minimal nonnegative integer c such that f(N)=N+c passes Axioms A (with A1 unit step)."""
    c = 0
    while True:
        f = lambda N: mapping_affine(N,c)
        ok, why = passes_axioms(f, Nmin=Nmin, Nmax=Nmax, reserved=reserved, require_step=1)
        if ok:
            return c, why
        c += 1

def mapping_table(mapping_fn, Nmin=2, Nmax=8):
    rows = []
    for N in range(Nmin, Nmax+1):
        inv = su_invariants(N)
        rows.append({
            "N": N,
            "f(N)": mapping_fn(N),
            "rank": inv["rank"],
            "dim": inv["dim"],
            "coxeter": inv["coxeter"],
            "pos_roots": inv["pos_roots"],
        })
    return rows
