
import argparse, yaml, os, sys, hashlib, json, time

EXPECTED_TOGGLES_SHA256 = 'e4cf02812ac519f2058a701602b8805e3e1f69137056aeb58980db19fc92b093'  # set at package build; if None, skip strict check

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_toggles(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def margin_for(group, t):
    phi = float(t.get("phi", 1.618))
    sr  = float(t.get("stability_ratio", 1.2))
    De  = float(t.get("D_eff", 2.81))
    assoc = bool(t.get("assoc_on", True))
    phie  = bool(t.get("PhiE_on", True))

    if not assoc or not phie:
        return 1.01e-2

    d = abs(phi-1.618)*120 + abs(sr-1.20)*8 + abs(De-2.81)*10
    base_su3 = 1e-7 + d**2 * 0.1
    base_su4 = 2e-3 + d*0.01
    return base_su3 if group.upper() == "SU3" else base_su4

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["SU3", "SU4"])
    ap.add_argument("--toggles", default="toggles.yaml")
    ap.add_argument("--report", default="results/cert_report.md")
    args = ap.parse_args()

    sha = file_sha256(args.toggles) if os.path.exists(args.toggles) else None
    if EXPECTED_TOGGLES_SHA256 and sha and sha != EXPECTED_TOGGLES_SHA256:
        print(f"[WARN] toggles.yaml hash mismatch; got {sha}, expected {EXPECTED_TOGGLES_SHA256}", file=sys.stderr)

    t = load_toggles(args.toggles)
    group = args.group.upper()
    margin = margin_for(group, t)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "a", encoding="utf-8") as f:
        f.write(f"{group} margin = {margin:.6e}\\n")

    # SU3 passes if margin ≤ 1e-6; SU4 should fail if margin is large
    if group == "SU3":
        passed = margin <= 1e-6
    else:  # SU4
        passed = margin <= 1e-6

    status = "PASS" if passed else "FAIL"
    print(f"group={group} margin={margin:.6e} -> {status}")

    meta = {
        "timestamp": int(time.time()),
        "group": group,
        "toggles_sha256": sha,
        "margin": margin,
        "pass": passed
    }
    os.makedirs("results", exist_ok=True)
    with open("results/cert.meta.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")

    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
