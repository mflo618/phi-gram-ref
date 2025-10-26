#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Traceability: real deterministic computation (no mocks), no network I/O.
# Deterministic Decimal arithmetic. Non-circular by construction (no alpha usage).
# See HOW_TO_VERIFY.md, MANIFEST.json, CHECKSUMS.txt.
# -----------------------------------------------------------------------------
import argparse, ast, json, os, hashlib, re
BANNED={'alpha','alpha_inv','fine_structure','fineStructure','alphaInverse'}
def hash_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()
def scan(p):
    src=open(p,'r',encoding='utf-8').read()
    hits=[]
    try:
        tree=ast.parse(src, filename=p)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in BANNED:
                hits.append({'kind':'ast_name','name':node.id,'lineno':getattr(node,'lineno',None)})
            if isinstance(node, ast.Attribute) and node.attr in BANNED:
                hits.append({'kind':'ast_attr','name':node.attr,'lineno':getattr(node,'lineno',None)})
    except SyntaxError as e:
        hits.append({'kind':'syntax_error','msg':str(e)})
    for b in BANNED:
        for m in re.finditer(rf'(?i)\b{re.escape(b)}\b', src):
            hits.append({'kind':'token','name':b,'span':[m.start(), m.end()]})
    return hits
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--paths', nargs='+', default=['lst_sigma_em0.py','g_electron.py','electron_density_certificate.py','density_certificate_v2.py'])
    ap.add_argument('--json')
    a=ap.parse_args()
    report={'files':[]}; bad=False
    for p in a.paths:
        if not os.path.exists(p):
            report['files'].append({'path':p,'error':'missing'}); continue
        h=hash_file(p); hits=scan(p); report['files'].append({'path':p,'sha256':h,'hits':hits})
        if any(hh['kind'].startswith('ast_') for hh in hits): bad=True
    print(json.dumps(report, indent=2))
    if a.json:
        with open(a.json,'w') as f: json.dump(report, f, indent=2)
    if bad: raise SystemExit(2)
