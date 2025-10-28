#!/usr/bin/env python3
import argparse, json, math, os, random, re, subprocess, sys
from pathlib import Path

# ---------- tiny helpers ----------
def load_config(p): return json.loads(Path(p).read_text())
def outdir(cfg): 
    d = Path(cfg.get("output_dir","grc_outputs")); d.mkdir(parents=True, exist_ok=True); return d
def run(cmd): return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout

METRIC_PATTERNS = {
    "rel_min": [r"rel[_ ]?min[:=]\s*([0-9.eE+-]+)", r"minNorm[:=]\s*([0-9.eE+-]+)"],
    "convergence_rate": [r"convergence[_ ]?rate[:=]\s*([0-9.eE+-]+)", r"conv(?:er)?gence[:=]\s*([0-9.eE+-]+)"],
    "rho_T": [r"rho\(T\)[:=]\s*([0-9.eE+-]+)", r"spectral[_ ]?radius.*?([0-9.eE+-]+)"],
    "lambda_min": [r"lambda[_ ]?min.*?([0-9.eE+-]+)", r"λ[_ ]?min.*?([0-9.eE+-]+)"],
    "lambda_max": [r"lambda[_ ]?max.*?([0-9.eE+-]+)", r"λ[_ ]?max.*?([0-9.eE+-]+)"],
    "cond": [r"cond(?:ition)?(?:ing)?(?:\(G\))?[:=]\s*([0-9.eE+-]+)"]
}
def parse_metrics(txt):
    def fkey(pats):
        for pat in pats:
            m=re.search(pat, txt, re.I)
            if m:
                try: return float(m.group(1))
                except: pass
        return None
    out = {k:fkey(v) for k,v in METRIC_PATTERNS.items()}
    if re.search(r"\bverdict\b.*\bPASS\b", txt, re.I): out["verdict"]="PASS"
    if re.search(r"\bverdict\b.*\bFAIL\b", txt, re.I): out["verdict"]="FAIL"
    return out

def su_pair(N): return (N+2, N+3)  # unused now; suite runs on {su3,su4}
def spearman(xs, ys):
    ranks = lambda a: {v:i+1 for i,v in enumerate(sorted(set(a)))}
    rx, ry = ranks(xs), ranks(ys)
    n=len(xs); 
    return None if n<3 else 1 - 6*sum((rx[x]-ry[y])**2 for x,y in zip(xs,ys))/(n*(n*n-1))

# ---------- gates ----------
def gate_offset(cfg, od):
    cmd = cfg["scanner_cmd"]; obj = cfg["objective"]
    a = cmd + ["--group","su4","--harmonics","5,6","--objective",obj,"--delta","0.0","--check-dynamics"]
    b = cmd + ["--group","su4","--harmonics","5,6","--objective",obj,"--delta",str(cfg.get("delta_offset",0.44879895)),"--check-dynamics"]
    A, B = run(a), run(b)
    mA, mB = parse_metrics(A), parse_metrics(B)
    res = {"group":"su4","harmonics":[5,6],"delta0":0.0,"delta1":cfg.get("delta_offset",0.44879895),
           "run0":{"stdout":A,"metrics":mA},"run1":{"stdout":B,"metrics":mB},
           "verdict_unchanged": (mA.get("verdict")==mB.get("verdict"))}
    (od/"offset_result.json").write_text(json.dumps(res, indent=2))
    print("Gate 1 (offset) →", od/"offset_result.json")

def gate_delta_scan(cfg, od):  # replaces gamma-scan
    cmd, obj, ds = cfg["scanner_cmd"], cfg["objective"], cfg["delta_values"]
    rows=[]
    for d in ds:
        r = run(cmd + ["--group","su3","--harmonics","5,6","--objective",obj,"--delta",str(d),"--check-dynamics"])
        rows.append({"delta":d,"metrics":parse_metrics(r),"stdout":r})
    verdicts = sorted({r["metrics"].get("verdict") for r in rows})
    j = {"group":"su3","harmonics":[5,6],"rows":rows,"verdicts":verdicts,"invariant":(len(verdicts)==1)}
    (od/"gamma_scan_su3_5_6.json").write_text(json.dumps(j, indent=2))  # keep filename for paper
    print("Gate 2 (δ-scan for invariance) →", od/"gamma_scan_su3_5_6.json")

def gate_null(cfg, od):
    cmd, obj = cfg["scanner_cmd"], cfg["objective"]
    scr = int(cfg["null"]["scrambles_per_pair"]); su = cfg["null"]["su_for_null"]
    rank_pairs = [tuple(p) for p in cfg["null"]["rank_pairs"]]
    cox_pairs  = [tuple(p) for p in cfg["null"]["coxeter_pairs"]]

    fam = {"rank":[], "coxeter":[]}
    for (m,M) in rank_pairs:
        for _ in range(scr):
            d=random.choice(cfg["delta_values"])
            r=run(cmd+["--group",su,"--harmonics",f"{m},{M}","--objective",obj,"--delta",str(d),"--check-dynamics"])
            met=parse_metrics(r); 
            if met.get("rel_min") and met.get("convergence_rate"):
                met.update({"pair":[m,M],"delta":d}); fam["rank"].append(met)
    for (m,M) in cox_pairs:
        for _ in range(scr):
            d=random.choice(cfg["delta_values"])
            r=run(cmd+["--group",su,"--harmonics",f"{m},{M}","--objective",obj,"--delta",str(d),"--check-dynamics"])
            met=parse_metrics(r); 
            if met.get("rel_min") and met.get("convergence_rate"):
                met.update({"pair":[m,M],"delta":d}); fam["coxeter"].append(met)

    alpha = float(cfg.get("alpha",0.01))
    union = fam["rank"] + fam["coxeter"]
    rel = [u["rel_min"] for u in union]
    dyn = [u["convergence_rate"] for u in union]

    def qtail(arr, q, lower=True):
        arr = sorted(arr)
        if not arr: return None
        return arr[max(0,int(math.floor(q*len(arr)))-1)] if lower else arr[min(len(arr)-1,int(math.ceil((1-q)*len(arr))))]

    rel_thr = qtail(rel, alpha, lower=True)
    dyn_thr = qtail(dyn, alpha, lower=False)

    lrel=[math.log(x) for x in rel] if rel else []
    ldyn=[math.log(x) for x in dyn] if dyn else []
    mean = (lambda a: sum(a)/len(a)) if lrel or ldyn else None
    def stats(a):
        if not a: return {"mu":None,"sigma":None,"n":0}
        mu=sum(a)/len(a); var=sum((x-mu)**2 for x in a)/len(a) if len(a)>1 else None
        return {"mu":mu,"sigma":(var**0.5 if var is not None else None),"n":len(a)}

    out = {"source":"union-null(rank+coxeter)","alpha":alpha,
           "thresholds":{"rel_min":rel_thr,"convergence_rate":dyn_thr},
           "summary":{"rel_min":stats(lrel),"convergence_rate":stats(ldyn)},
           "counts":{"rank":len(fam["rank"]),"coxeter":len(fam["coxeter"]),"union":len(union)}}
    (od/"null_thresholds.json").write_text(json.dumps(out, indent=2))
    print("Gate 3 (null thresholds) →", od/"null_thresholds.json")

def gate_su_suite(cfg, od):
    j = json.loads((od/"null_thresholds.json").read_text())
    t_rel, t_dyn = j["thresholds"]["rel_min"], j["thresholds"]["convergence_rate"]
    mu_rel, sig_rel = j["summary"]["rel_min"]["mu"], j["summary"]["rel_min"]["sigma"]
    mu_dyn, sig_dyn = j["summary"]["convergence_rate"]["mu"], j["summary"]["convergence_rate"]["sigma"]

    cmd, obj = cfg["scanner_cmd"], cfg["objective"]
    rows=[]; labels=[]
    for G in cfg["su_groups_for_suite"]:
        # use canonical pairs for the paper: su3→(5,6), su4→(6,7)
        pair = (5,6) if G=="su3" else (6,7)
        r=run(cmd+["--group",G,"--harmonics",f"{pair[0]},{pair[1]}","--objective",obj,"--delta","0.0","--check-dynamics"])
        met=parse_metrics(r)
        rel, dyn = met.get("rel_min"), met.get("convergence_rate")
        z_static = ( (mu_rel - math.log(rel))/sig_rel ) if (rel and mu_rel and sig_rel) else None
        z_dynamic= ( (mu_dyn - math.log(dyn))/sig_dyn ) if (dyn and mu_dyn and sig_dyn) else None
        access = (z_static - z_dynamic) if (z_static is not None and z_dynamic is not None) else None
        verdict = None
        if all(v is not None for v in [rel,dyn,t_rel,t_dyn]): verdict = "PASS" if (rel<=t_rel and dyn<=t_dyn) else "FAIL"
        rows.append({"group":G,"harmonics":list(pair),"rel_min":rel,"convergence_rate":dyn,
                     "z_static":z_static,"z_dynamic":z_dynamic,"accessibility":access,"verdict":verdict,"stdout":r})
        if z_dynamic is not None: labels.append((G,z_dynamic))

    # With only two groups, monotonicity over N isn’t defined → set None.
    outj = {"thresholds":{"rel_min":t_rel,"convergence_rate":t_dyn},
            "rows": rows, "spearman_dynamic_z_vs_N": None}
    (od/"su_suite_results.json").write_text(json.dumps(outj, indent=2))
    print("Gate 4 (SU3/SU4 suite) →", od/"su_suite_results.json")

def gate_blinded_min(cfg, od):
    thr = json.loads((od/"null_thresholds.json").read_text())
    t_rel, t_dyn = thr["thresholds"]["rel_min"], thr["thresholds"]["convergence_rate"]
    cmd, obj = cfg["scanner_cmd"], cfg["objective"]
    labels=["A","B"]; groups=["su3","su4"]; random.seed(42); random.shuffle(labels)
    mask = dict(zip(labels, groups))
    runs=[]
    for L in labels:
        G = mask[L]; pair = (5,6) if G=="su3" else (6,7)
        r=run(cmd+["--group",G,"--harmonics",f"{pair[0]},{pair[1]}","--objective",obj,"--delta","0.0","--check-dynamics"])
        met=parse_metrics(r); rel, dyn = met.get("rel_min"), met.get("convergence_rate")
        verdict = None
        if all(v is not None for v in [rel,dyn,t_rel,t_dyn]): verdict = "PASS" if (rel<=t_rel and dyn<=t_dyn) else "FAIL"
        runs.append({"label":L,"metrics":met,"verdict":verdict,"stdout":r})
    (od/"blinded_min.json").write_text(json.dumps({"masked_runs":runs,"unmask":mask}, indent=2))
    print("Gate 5 (blinded SU3↔SU4) →", od/"blinded_min.json")

# ---------- main ----------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()
    cfg=load_config(args.config); od=outdir(cfg)

    gate_offset(cfg, od)
    gate_delta_scan(cfg, od)
    gate_null(cfg, od)
    gate_su_suite(cfg, od)
    gate_blinded_min(cfg, od)

    print("\nAll minimal gates complete. See:", od, "\n")
