import csv, re, sys
CHAIN = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")
def main(expdir, min_iter=2000):
    n=junk=prefab=deep=0; pay=jpay=ppay=dpay=0.0
    with open(f'{expdir}/reproduction_report.csv') as f:
        rd = csv.reader(f, delimiter=';')
        next(rd)
        for r in rd:
            try:
                t=int(r[0]); tok=float(r[4]); sc=float(r[5])
            except (ValueError, IndexError): continue
            if t<min_iter or sc<=0 or len(r)<9 or 'OrderedDict' not in r[8]: continue
            parts = CHAIN.findall(r[8])
            n+=1; pay+=max(tok,0)
            if sc<=0.05: junk+=1; jpay+=max(tok,0)
            if any(p.startswith(('clusterer_prefab','vectorSpace_prefab')) for p in parts): prefab+=1; ppay+=max(tok,0)
            if len(parts)>6: deep+=1; dpay+=max(tok,0)
    print(f'{expdir}')
    print(f'  settles {n}, total pay {pay:.0f}')
    print(f'  JUNK (score<=0.05):    {junk/n:6.1%} of settles, {jpay/max(pay,1e-9):6.1%} of pay')
    print(f'  PREFAB-anchored:       {prefab/n:6.1%} of settles, {ppay/max(pay,1e-9):6.1%} of pay')
    print(f'  DEEP (>6 parts):       {deep/n:6.1%} of settles, {dpay/max(pay,1e-9):6.1%} of pay')
main(sys.argv[1])
