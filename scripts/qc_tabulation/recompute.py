import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1], low_memory=False)
f = lambda c: pd.to_numeric(df[c], errors='coerce').fillna(0)
# exclude special calc units (MFIP, pure PA etc.) flagged by Tech Doc
special = (f('MN_FIP')==1)
d = df[~special].copy()
g = lambda c: pd.to_numeric(d[c], errors='coerce').fillna(0)
size = g('FSUSIZE')
# L1: federal benefit formula from QC's own net income and max benefit
calc = g('BENMAX') - np.ceil(0.3*g('FSNETINC'))
calc_r = g('BENMAX') - np.round(0.3*g('FSNETINC'))
minb = g('FSMINBEN').where(g('FSMINBEN')>0, g('MINIMUM_BEN'))
def ben(c): 
    return np.where(size<=2, np.maximum(minb, c), np.maximum(0, c))
for name,c in [('ceil',calc),('round',calc_r)]:
    b = ben(c); diff = np.abs(b - g('FSBEN'))
    print(f"L1 benefit formula ({name}): exact {np.mean(diff==0):.4f}, within $1 {np.mean(diff<=1):.4f}, within $5 {np.mean(diff<=5):.4f}")
# L2: net income from components
ded = g('FSSTDDED')+g('FSERNDED')+g('FSDEPDED')+g('FSMEDDED')+g('FSCSDED')+g('FSSLTDED')
net = np.maximum(0, g('FSGRINC')-ded)
print("L2 FSTOTDED = sum of listed deductions: within $1", round(np.mean(np.abs(ded-g('FSTOTDED'))<=1),4))
print("L2 FSNETINC from FSGRINC-FSTOTDED: within $1", round(np.mean(np.abs(np.maximum(0,g('FSGRINC')-g('FSTOTDED'))-g('FSNETINC'))<=1),4))
# L3: shelter deduction from expenses and cap
pre = g('FSGRINC') - (g('FSSTDDED')+g('FSERNDED')+g('FSDEPDED')+g('FSMEDDED')+g('FSCSDED'))
excess = np.maximum(0, g('FSSLTEXP') - np.floor(0.5*np.maximum(pre,0)))
exempt = (g('FSELDER')>0)|(g('FSNDIS')>0)
sh = np.where(exempt, excess, np.minimum(excess, g('SHELCAP')))
dsh = np.abs(sh - g('FSSLTDED'))
print("L3 shelter deduction from FSSLTEXP & SHELCAP: within $1", round(np.mean(dsh<=1),4), " within $5", round(np.mean(dsh<=5),4))
# L3b: FSSLTEXP = RENT + UTIL
print("L3b FSSLTEXP = RENT+UTIL within $1:", round(np.mean(np.abs(g('RENT')+g('UTIL')-g('FSSLTEXP'))<=1),4))
# FSBEN vs RAWBEN among cases QC found correct
ok = d.STATUS==1
print("FSBEN == RAWBEN where STATUS=1 (within $1):", round(np.mean(np.abs(g('FSBEN')-g('RAWBEN'))[ok]<=1),4), " n=", int(ok.sum()))
# per-state agreement of full chain L1 (best variant)
b = ben(calc); d['ok1'] = np.abs(b-g('FSBEN'))<=1; d['ok3']= dsh<=1
st = d.groupby('STATENAME')[['ok1','ok3']].mean().sort_values('ok1')
print("\nper-state L1 within $1: min", round(st.ok1.min(),3), st.ok1.idxmin(), " median", round(st.ok1.median(),3))
print("per-state L3 within $1: min", round(st.ok3.min(),3), st.ok3.idxmin(), " median", round(st.ok3.median(),3))
print(st.head(6).round(3).to_string())
print("Michigan:", st.loc['Michigan'].round(3).to_dict())
# state parameters visible in data
print("\nBENMAX for size 1..4 (48 states) distinct values:", {s: sorted(pd.unique(d.loc[(size==s)&(~d.STATENAME.isin(['Alaska','Hawaii','Guam','Virgin Islands'])),'BENMAX']))[:4] for s in [1,2,3,4]})
print("SHELCAP distinct (non-AK/HI/GU/VI):", sorted(pd.unique(d.loc[~d.STATENAME.isin(['Alaska','Hawaii','Guam','Virgin Islands']),'SHELCAP']))[:6])
print("FSSTDDED by size (48 states):", {s: sorted(pd.unique(d.loc[(size==s)&(~d.STATENAME.isin(['Alaska','Hawaii','Guam','Virgin Islands'])),'FSSTDDED']))[:4] for s in [1,3,5,6]})
