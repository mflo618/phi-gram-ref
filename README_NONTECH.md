# PHI‑GRAM: Quick Guide (Non‑Technical)

This little tool runs a **fair test** that compares two patterns called **SU(3)** and **SU(4)**.
You don’t need to know the math—just read the traffic‑light style summary it prints.

---

## 1) One‑minute use

1) Open a terminal in the folder with `phi_gram_ref.py`  
2) Run this command (copy/paste):

```
python phi_gram_ref.py --cesaro-start-T --check-dynamics --group both
```

That’s it. You’ll see two blocks labeled **SU3 CERTIFICATE** and **SU4 CERTIFICATE**.

> Tip: If your terminal doesn’t show color/emoji, add `--no-color` at the end.

---

## 2) What the verdict means

At the end of each block you’ll see a **Verdict** line, green/yellow/red:
- **Coherent resonance** (green): the system “locks in” to one direction.
- **Weak resonance** (yellow): partial lock‑in.
- **Geometric obstruction** (red): it **doesn’t** lock in.

You’ll also see a bar like `[███████░░░░░░░░░░]`. More **filled** = more coherent.

**Two things to glance at:**
- `rel_min` (printed with a ✅/❌) — smaller is better (more coherence).  
  It’s defined as:

  `rel_min = min‖u + h·v‖² / tr(G)`

- “Cesàro stability” (tiny number is good) — confirms the result is numerically stable.

**Rules of thumb used for the verdict**
- `rel_min ≤ 0.03` ⇒ **Coherent** (green)
- `0.03 < rel_min ≤ 0.15` ⇒ **Weak** (yellow)
- `rel_min > 0.15` ⇒ **Obstruction** (red)

---

## 3) The bridge (optional, one extra step)

If you also want an **end‑to‑end** check that connects the green/yellow/red result
to a simple one‑loop prediction, run:

```
python phi_gram_ref.py --cesaro-start-T --check-dynamics --group both --bridge --R0_SU3 1.152 --R0_SU4 1.20
```

You’ll see a **One‑Loop Bridge** block that prints a predicted value for SU(4).
That’s just a compact way of saying: “Using SU(3) as a calibration, the SU(4) number
should be about X.”

> You can also add `--bridge-agg median_topk` to make the prediction use the
> median of the best points, which is a robustness check.

---

## 4) Ablation (optional: a sanity check)

Want to verify the construction is essential and not a gimmick?
Run a **comparison** where we remove one safety step (the √Jacobian weight):

```
python phi_gram_ref.py --cesaro-start-T --check-dynamics --group both --compare-ablation
```

You’ll get an extra line showing how the key numbers change when that safety step is turned **off**.
This makes the test more credible: you can see the method gets **worse** without that guardrail.

---

## 5) What makes this a fair test?

- **Same pipeline for both** SU(3) and SU(4) — no hidden tweaks.
- **Deterministic** — same input gives the same output.
- **Multiple signals** — not just one number; you see stability checks and dynamics too.
- **Transparency** — the script can list the top‑K best points and save all the raw numbers
  in `phi_gram_summary.json` and `scan_*.json` for anyone to audit.

---

## 6) Typical outcomes (what you should expect)

- **SU(3)** → green “Coherent resonance” with a small `rel_min`, a short, full‑looking bar,
  and tiny stability error.
- **SU(4)** → red “Geometric obstruction” with a bigger `rel_min`, a shorter bar,
  and (usually) a faster dynamic rate.

If you run the **bridge**, you’ll also see a predicted SU(4) value that uses
the measured “speed” (convergence rate) of each case.

---

## 7) Troubleshooting

- **No color/emoji?** Add `--no-color`.
- **Slow run?** Reduce resolution: `--B 1024 --Nproj 48` (these are the defaults).
- **More accuracy?** Increase resolution: `--B 4096 --Nproj 96 --grid 36 --refine 2 --refine-grid 16`.
- **Plain text only?** Always add `--no-color` for logs.
- **Want all results in one file?** Check `phi_gram_summary.json` after the run.

---

## 8) FAQ (no math degree required)

**Q: What do SU(3) and SU(4) mean here?**  
A: Just two different symmetry patterns. The tool checks which one “locks in”
more strongly under the same conditions.

**Q: What are `u` and `v`?**  
A: Think of them as two directions we track. If the system is coherent,
they collapse to essentially one direction.

**Q: What if my numbers differ slightly from someone else’s?**  
A: Small differences can come from resolution settings (`B`, `Nproj`) and grid refinement.
The **color verdict** and the big picture should be the same.

**Q: Is the result cherry‑picked?**  
A: No. We use the **same** steps and thresholds for both cases, show stability checks,
and include transparency outputs so others can re‑check everything.

---

## 9) One‑line summary

If the SU(3) block is **green** and the SU(4) block is **red**, the test passed:
SU(3) behaves coherently; SU(4) does not—just like the paper says.
