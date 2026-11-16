import sys, pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False, usecols=['STATENAME','SUA1','SUA2','UTIL','FSUSIZE'])
d = df[df.SUA2==1]
hc = d[d.SUA1.isin([3,4,8])].groupby('STATENAME').UTIL.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
lua = d[d.SUA1==5].groupby('STATENAME').UTIL.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
share = df.groupby('STATENAME').SUA1.value_counts(normalize=True).unstack().fillna(0).round(2)
out = pd.DataFrame({'HCSUA_mode':hc,'LUA_mode':lua}).join(share[[c for c in [2,3,4,5,8] if c in share]])
print(out.loc[['Michigan','California','Texas','Ohio','Florida','Georgia','Illinois','Pennsylvania','Massachusetts','Mississippi']].to_string())
print("\nshare of households using actual utility expenses (SUA1=2), median state:", share[2].median() if 2 in share else 'n/a', " max:", share[2].max() if 2 in share else '', share[2].idxmax() if 2 in share else '')
