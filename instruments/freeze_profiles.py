import csv, re, collections, json
CHAIN = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")
def census(rep):
    rows = []
    for r in csv.reader(open(rep, errors="replace"), delimiter=";"):
        try: t = int(r[0]); sc = float(r[5])
        except (ValueError, IndexError): continue
        if sc <= 0 or len(r) < 9 or "OrderedDict" not in r[8]: continue
        rows.append((t, r[8]))
    tmax = max(t for t, _ in rows)
    tcut = tmax - max(500, tmax // 4)
    cnt = collections.Counter()
    n = 0
    for t, ch in rows:
        if t >= tcut:
            n += 1
            for p in set(CHAIN.findall(ch)): cnt[p] += 1
    return {"t_max": tmax, "t_cut": tcut, "n_deliveries": n, "counts": dict(cnt)}
frozen = {"source_X_bigtree_basic_seed227": census("./merge_work/bigtree_basic_seed227/cur.csv"),
          "source_Y_bigtree_alwayson_seed211": census("./merge_work/bigtree_alwayson_seed211/era.csv")}
json.dump(frozen, open("truth/cr3_frozen_profiles.json", "w"), indent=1)
for k, v in frozen.items():
    print("FROZEN", k, "t_max", v["t_max"], "deliveries", v["n_deliveries"], "tools", len(v["counts"]))
