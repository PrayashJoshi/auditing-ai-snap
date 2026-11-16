import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1], low_memory=False)
g = lambda c: pd.to_numeric(df[c], errors='coerce').fillna(0)
print("MINIMUM_BEN values:", df.MINIMUM_BEN.value_counts().head(5).to_dict(), " FSMINBEN values:", df.FSMINBEN.value_counts().head(5).to_dict())
size=g('FSUSIZE')
nonspecial = (g('MN_FIP')==0)&(g('CASE')!=3)
c = g('BENMAX') - np.ceil(0.3*g('FSNETINC'))
for minb in [23]:
    b = np.where(size<=2, np.maximum(minb,c), np.maximum(0,c))
    diff = (g('FSBEN') - b)[nonspecial]; ok = np.abs(diff)<=1
    print(f"L1 with min benefit ${minb}, excl. MFIP/demo: within $1 = {ok.mean():.4f} (n={int(nonspecial.sum())})")
    d = df[nonspecial].assign(ok=ok.values)
    st = d.groupby('STATENAME').ok.mean().sort_values()
    print("per-state: min", round(st.min(),3), st.idxmin(), " 5th worst", st.round(3).head(5).to_dict(), " median", round(st.median(),3), " Michigan", round(st['Michigan'],3), " CA", round(st['California'],3), " TX", round(st['Texas'],3))
    bad = ~ok
    print("remaining mismatch n:", int(bad.sum()), " diff quantiles:", pd.Series(diff[bad.values]).quantile([.1,.5,.9]).to_dict())
    print("remaining by size>2 & calc<=0:", int(((size[nonspecial]>2)&(c[nonspecial]<=0)&bad.values).sum()), " by MED_DED_DEMO=1:", int((g('MED_DED_DEMO')[nonspecial]==1)[bad.values].sum()))
