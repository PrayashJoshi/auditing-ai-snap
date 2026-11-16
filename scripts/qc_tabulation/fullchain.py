import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1], low_memory=False)
g = lambda c: pd.to_numeric(df[c], errors='coerce').fillna(0)
special = (g('MN_FIP')==1)|(g('CASE')==3)|(df.SUA1.isna()&df.FSSLTDED.isna())
size=g('FSUSIZE')
# calculator from application-level inputs + state/federal parameters
earn = g('FSEARN'); gross = g('FSGRINC')
erded = np.round(0.2*earn)                      # earned income deduction 20%
other = g('FSSTDDED') + erded + g('FSDEPDED') + g('FSMEDDED') + g('FSCSDED')
adj = np.maximum(0, gross - other)
shexp = g('RENT') + g('UTIL')                    # UTIL = state SUA when SUA used
excess = np.maximum(0, shexp - np.floor(0.5*adj))
exempt = (g('FSELDER')>0)|(g('FSNDIS')>0)
shel = np.where(exempt, excess, np.minimum(excess, g('SHELCAP')))
homeless = g('HOMEDED')>0
net = np.maximum(0, adj - shel)
c = g('BENMAX') - np.ceil(0.3*net)
ben = np.where(size<=2, np.maximum(g('MINIMUM_BEN'), c), np.maximum(0,c))
ok = np.abs(ben - g('FSBEN'))<=1
m = ~special
print("full chain (income+expenses -> benefit) within $1 of FSBEN, excl special:", round(ok[m].mean(),4), " n=", int(m.sum()))
print("  excl special and homeless-deduction units:", round(ok[m & ~homeless].mean(),4))
print("  earned ded matches FSERNDED within $1:", round((np.abs(erded-g('FSERNDED'))<=1)[m].mean(),4))
st = pd.Series(ok[m]).groupby(df.STATENAME[m].values).mean()
print("  per-state min:", round(st.min(),3), st.idxmin(), " median:", round(st.median(),3), " Michigan:", round(st['Michigan'],3))
bad = m & ~ok
print("  mismatches:", int(bad.sum()), " median |diff|:", float(np.median(np.abs(ben-g('FSBEN'))[bad])) if bad.any() else 0)
