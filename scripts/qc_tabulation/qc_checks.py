import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1], low_memory=False)
print("rows, cols:", df.shape)
need = ['STATE','STATENAME','STATUS','AMTERR','AUTHREP','ACTNTYPE','RAWBEN','FSBEN','HWGT','FYWGT','CASE']
print("missing key cols:", [c for c in need if c not in df.columns])
el = [f'ELEMENT{i}' for i in range(1,10)]
print("error-var cols present:", sum(c in df.columns for c in el), [c for c in df.columns if c.startswith('AMOUNT')][:3], [c for c in df.columns if c.startswith('OCCDAT')][:2])
st = df.groupby('STATENAME').size().sort_values(ascending=False)
print("\nstates:", len(st)); print(st.describe().round(0).to_string())
print("\nMichigan n:", int(st.get('Michigan', 0)), " California:", int(st.get('California',0)), " Texas:", int(st.get('Texas',0)))
print("\nSTATUS counts:", df.STATUS.value_counts().sort_index().to_dict())
df['err_over_tol'] = df.AMTERR > 56
print("share AMTERR>56:", round(df.err_over_tol.mean(),4), " weighted:", round(np.average(df.err_over_tol, weights=df.HWGT),4))
print("AUTHREP counts:", df.AUTHREP.value_counts(dropna=False).to_dict())
print("ACTNTYPE counts:", df.ACTNTYPE.value_counts(dropna=False).to_dict())
nvar = df[el].notna().sum(axis=1) if df[el].dtypes.iloc[0]!=object else (df[el].fillna('').astype(str).str.strip()!='').sum(axis=1)
errcase = df.STATUS.isin([2,3])
print("\nerror cases:", int(errcase.sum()), " #variances among error cases:", nvar[errcase].value_counts().sort_index().to_dict())
ag = df[[f'AGENCY{i}' for i in range(1,10)]]
client = ag.isin([1,2,3,4,7]).any(axis=1); agency = ag.apply(lambda s: s.between(8,99)).any(axis=1)
print("error cases with any client-caused variance:", int((errcase&client).sum()), " any agency-caused:", int((errcase&agency).sum()), " client-only:", int((errcase&client&~agency).sum()))
tp = df[[f'TIMEPER{i}' for i in range(1,10)]]
filing = tp.isin([1,2]).any(axis=1) & ~tp.isin([3]).any(axis=1)
single_client_filing = errcase & (nvar==1) & client & filing
print("single-variance, client-caused, at/before action (cleanly rebuildable human arm):", int(single_client_filing.sum()))
for s in ['Michigan','California','Texas']:
    m = df.STATENAME==s
    print(f"  {s}: n={int(m.sum())}, errors={int((m&errcase).sum())}, >56={int((m&df.err_over_tol).sum())}, AUTHREP=1: {int((m&(df.AUTHREP==1)).sum())}, rebuildable={int((m&single_client_filing).sum())}")
print("\nper-state rebuildable (min/median/max):", single_client_filing.groupby(df.STATENAME).sum().describe()[['min','50%','max']].to_dict())
