## 2) `PROTOCOL.md`
```md
# GRC Falsification Cascade — Minimal (Paper)

Fixed choices:
- Objective: minimize `rel_min`.
- Mapping: SU(N) → (N+2, N+3).
- γ policy: test γ ∈ {0, π/4, π/2, 3π/4, π}; if verdict invariant, fix γ=0 thereafter.
- Seeds/grid are your scanner defaults (the pack doesn’t touch them).

Gates executed here:
1) Offset removal (SU4 @ (5,6)): δ=0 vs δ=π/7; verdict unchanged ⇒ offset removed.
2) γ-invariance (SU3 @ (5,6)): invariant verdict ⇒ γ fixed to 0 for the rest.
3) Null thresholds (union of rank & Coxeter families; 10 scrambles each; α=1% tails).
4) Monotonicity (SU2–SU6): Spearman ρ on dynamic_z vs N.
5) Minimal blinded check (SU2–SU4) with thresholds frozen.

Artifacts are signed by commit SHA via your VCS (include SHA in the paper’s ledger).
````

---
