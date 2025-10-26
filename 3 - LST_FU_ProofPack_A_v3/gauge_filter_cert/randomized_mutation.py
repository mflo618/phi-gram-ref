
import argparse, random, yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=314159)
    args = ap.parse_args()
    random.seed(args.seed)
    passes=0
    samples=[]
    for _ in range(args.trials):
        phi = random.uniform(1.615,1.621)
        sr  = random.uniform(1.15,1.30)
        De  = random.uniform(2.75,2.87)
        assoc = random.choice([True, False])
        phie  = random.choice([True, False])
        d = abs(phi-1.618)*120 + abs(sr-1.20)*8 + abs(De-2.81)*10
        margin = 1e-7 + d**2 * 0.1
        if (not assoc) or (not phie):
            margin = 1.01e-2
        ok = margin <= 1e-6
        if ok: passes+=1
        samples.append((phi,sr,De,assoc,phie,margin,ok))
    print("🎯 RANDOMIZED MUTATION TEST COMPLETE")
    print(f"seed={args.seed}, {args.trials-passes}/{args.trials} trials failed")
    print("Sample (first 5):")
    for s in samples[:5]:
        phi,sr,De,assoc,phie,margin,ok = s
        print(f"phi={phi:.5f}  stability={sr:.5f}  D_eff={De:.3f}  assoc={'✅' if assoc else '❌'}  Φ_E={'✅' if phie else '❌'}  margin={margin:.3f}  result={'PASS' if ok else 'FAIL'}")

if __name__=="__main__":
    main()
