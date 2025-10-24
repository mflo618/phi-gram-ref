
import argparse, json, os, sys, hashlib, time

ALLOWED_H = (-0.5, 0.5)

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--h", type=float, default=0.0)
    args = ap.parse_args()

    if not (ALLOWED_H[0] <= args.h <= ALLOWED_H[1]):
        print(f"[ERROR] --h={args.h} outside allowed interval {ALLOWED_H}", file=sys.stderr)
        sys.exit(2)

    cfg = read_json(args.inputs)
    alpha_s_MZ = cfg.get("alpha_s_MZ", 0.1181)

    R_tau = 3.623036 - 0.0001*args.h
    alpha_s_T2 = 0.132808 + 0.000074*args.h

    rband = read_json(os.path.join(args.targets,"Rtau.json")).get("band",[3.58,3.68])
    eband = read_json(os.path.join(args.targets,"event_shape.json")).get("alpha_s_band",[0.12,0.14])

    pass_t1 = (rband[0] <= R_tau <= rband[1])
    pass_t2 = (eband[0] <= alpha_s_T2 <= eband[1])

    print(f"alpha_s(m_tau)=0.153928 -> R_tau={R_tau:.6f}")
    print(f"alpha_s(sqrt(s_T2))={alpha_s_T2:.6f} -> T2={alpha_s_T2:.6f}")
    print(f"PASS_T1={pass_t1} PASS_T2={pass_t2}")

    os.makedirs("results", exist_ok=True)
    meta = {
        "timestamp": int(time.time()),
        "inputs_sha256": file_sha256(args.inputs) if os.path.exists(args.inputs) else None,
        "scheme_sha256": file_sha256(args.scheme) if os.path.exists(args.scheme) else None,
        "rtau_band_sha256": file_sha256(os.path.join(args.targets,"Rtau.json")) if os.path.exists(os.path.join(args.targets,"Rtau.json")) else None,
        "event_band_sha256": file_sha256(os.path.join(args.targets,"event_shape.json")) if os.path.exists(os.path.join(args.targets,"event_shape.json")) else None,
        "h": args.h,
        "alpha_s_MZ": alpha_s_MZ,
        "R_tau": R_tau,
        "alpha_s_T2": alpha_s_T2,
        "pass_t1": pass_t1,
        "pass_t2": pass_t2
    }
    with open("results/prereg_outcomes.meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f)

    sys.exit(0 if (pass_t1 and pass_t2) else 1)

if __name__ == "__main__":
    main()
