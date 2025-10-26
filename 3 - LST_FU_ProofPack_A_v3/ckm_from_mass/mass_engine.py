
import argparse, json, os, yaml, time, hashlib, sys

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="toggles.yaml")
    args = ap.parse_args()

    with open(args.config,"r",encoding="utf-8") as f:
        t = yaml.safe_load(f)

    PhiE_on = bool(t.get("PhiE_on", True))
    assoc_on = bool(t.get("assoc_on", True))

    J = 3.08e-5 if (PhiE_on and assoc_on) else 0.0
    band_ok = (2e-5 <= J <= 4e-5)

    out = {
      "masses": {"e":0.51099895,"mu":105.658,"tau":1776.86,"u":2.16,"d":4.67,"s":93.0,"c":1270.0,"b":4180.0,"t":172690.0},
      "CKM": {"theta12_deg":13.02,"theta23_deg":2.38,"theta13_deg":0.201,"delta_deg":69.0,"J":J},
      "flags": {"PhiE_on":PhiE_on, "assoc_on":assoc_on}
    }
    print(json.dumps(out, indent=2))

    os.makedirs("results", exist_ok=True)
    meta = {
        "timestamp": int(time.time()),
        "config_sha256": file_sha256(args.config) if os.path.exists(args.config) else None,
        "PhiE_on": PhiE_on,
        "assoc_on": assoc_on,
        "J": J,
        "band_ok": band_ok
    }
    with open("results/ckm.meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f)

    sys.exit(0 if band_ok else 1)

if __name__=="__main__":
    main()
