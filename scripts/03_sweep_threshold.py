import geopandas as gpd, pandas as pd
g = gpd.read_file("data/processed/labels.geojson")
rows = []
for t in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    flag = (g["slum_area_frac"] > t).astype(int)
    rec = {"ambang": t}
    for city, sub in g.groupby("city"):
        s = flag[sub.index]
        rec[city] = f"{int(s.sum()):>3}/{len(sub)} ({s.mean():.0%})"
    rec["TOTAL"] = f"{int(flag.sum())}/{len(g)} ({flag.mean():.0%})"
    rows.append(rec)
print(pd.DataFrame(rows).to_string(index=False))