"""One-line society-ness reader: is a real society forming?
Reads: distinct learner specializations + spread, junk share,
prefab share, deep share, settle rate. Usage: societyness.py <expdir> [min_iter]"""
import csv, re, sys, collections
CHAIN = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")
TOOL = ('clusterer','vectorSpace','anomalyDetector','preprocessor','classifier','labeler','nearestNeighbors')
def main(expdir, min_iter=0):
    per=collections.defaultdict(collections.Counter); n=junk=prefab=deep=0; last=0
    with open(f'{expdir}/reproduction_report.csv') as f:
        rd=csv.reader(f,delimiter=';'); next(rd)
        for r in rd:
            try: t=int(r[0]); a=int(r[1]); sc=float(r[5])
            except (ValueError,IndexError): continue
            last=max(last,t)
            if t<min_iter or sc<=0 or len(r)<9 or 'OrderedDict' not in r[8]: continue
            parts=CHAIN.findall(r[8]); n+=1
            if sc<=0.05: junk+=1
            if any(p.startswith(('clusterer_prefab','vectorSpace_prefab')) for p in parts): prefab+=1
            if len(parts)>6: deep+=1
            for bare in set(parts):
                if bare.startswith(TOOL) and a>=6: per[a][bare]+=1
    specs=collections.Counter()
    for a,c in per.items():
        if c: specs[c.most_common(1)[0][0]]+=1
    n_specs=len(specs); n_learners=sum(specs.values())
    even=0.0
    if n_specs>1:
        import math
        tot=sum(specs.values())
        even=-sum((v/tot)*math.log2(v/tot) for v in specs.values())/math.log2(n_specs)
    top=', '.join(f'{k}:{v}' for k,v in specs.most_common(4))
    print(f'{expdir.split("/")[-1]}: age {last}, settles {n} ({n/max(last,1):.2f}/iter)')
    print(f'  DIVISION: {n_specs} distinct specializations over {n_learners} learners (evenness {even:.2f}) -> {top}')
    print(f'  junk {junk/max(n,1):.0%} | prefab {prefab/max(n,1):.0%} | deep {deep/max(n,1):.0%}')
main(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 0)
