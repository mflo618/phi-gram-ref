#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Traceability: real deterministic computation (no mocks), no network I/O.
# Deterministic Decimal arithmetic. Non-circular by construction (no alpha usage).
# See HOW_TO_VERIFY.md, MANIFEST.json, CHECKSUMS.txt.
# -----------------------------------------------------------------------------
from decimal import Decimal, getcontext
getcontext().prec = 80
import argparse, json, math
from density_certificate_v2 import alpha_inv_from_scale
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--scale', required=True)
    ap.add_argument('--rel-span', default='0.05')
    ap.add_argument('--points', type=int, default=9)
    ap.add_argument('--json')
    a=ap.parse_args()
    s0=Decimal(a.scale); span=Decimal(a.rel_span); n=max(3,int(a.points))
    rows=[]; xs=[]; ys=[]
    for i in range(n):
        t=Decimal(i)/Decimal(n-1)
        s=s0*(Decimal(1)-span+(Decimal(2)*span)*t)
        av=alpha_inv_from_scale(s)
        rows.append({'scale':str(s),'alpha_inv':str(av)})
        xs.append(math.log(float(s))); ys.append(math.log(float(av)))
    xm=sum(xs)/len(xs); ym=sum(ys)/len(ys)
    num=sum((x-xm)*(y-ym) for x,y in zip(xs,ys)); den=sum((x-xm)**2 for x in xs)
    m=num/den if den!=0 else float('nan'); b=ym-m*xm
    out={'slope_loglog':m,'intercept':b,'expected_slope':2.0,'rows':rows}
    print(json.dumps(out, indent=2))
    if a.json:
        with open(a.json,'w') as f: json.dump(out, f, indent=2)
