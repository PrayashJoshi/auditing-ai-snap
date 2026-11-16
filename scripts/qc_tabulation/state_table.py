import sys, re, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1], low_memory=False)
g = lambda c: pd.to_numeric(df[c], errors='coerce')
size=g('FSUSIZE').fillna(0); minb=g('MINIMUM_BEN').fillna(23)
c = g('BENMAX').fillna(0) - np.ceil(0.3*g('FSNETINC').fillna(0))
b = np.where(size<=2, np.maximum(minb,c), np.maximum(0,c))
special = (g('MN_FIP')==1)|(g('CASE')==3)|g('SUA1').isna()&g('FSSLTDED').isna()
df['formula_ok'] = (np.abs(g('FSBEN')-b)<=1)
df['special_calc'] = special
el=[f'ELEMENT{i}' for i in range(1,10)]; ag=df[[f'AGENCY{i}' for i in range(1,10)]]; tp=df[[f'TIMEPER{i}' for i in range(1,10)]]
nvar=df[el].notna().sum(axis=1); err=df.STATUS.isin([2,3]); client=ag.isin([1,2,3,4,7]).any(axis=1)
filing=tp.isin([1,2]).any(axis=1)&~tp.isin([3]).any(axis=1)
df['rebuildable']=err&(nvar==1)&client&filing
mode=lambda s: s.mode().iloc[0] if len(s.dropna()) else np.nan
d=df[df.SUA2==1]
t = pd.DataFrame({
 'n': df.groupby('STATENAME').size(),
 'errors_any': err.groupby(df.STATENAME).sum(),
 'errors_over_56': (g('AMTERR')>56).groupby(df.STATENAME).sum(),
 'authrep_1': (g('AUTHREP')==1).groupby(df.STATENAME).sum(),
 'rebuildable_client_errors': df.groupby('STATENAME').rebuildable.sum(),
 'HCSUA_mode': d[d.SUA1.isin([3,4,8])].groupby('STATENAME').UTIL.agg(mode),
 'LUA_mode': d[d.SUA1==5].groupby('STATENAME').UTIL.agg(mode),
 'TUA_mode': d[d.SUA1==6].groupby('STATENAME').UTIL.agg(mode),
 'share_actual_utility_expenses': (df.SUA1==2).groupby(df.STATENAME).mean().round(3),
 'share_cat_elig': (g('CAT_ELIG')>0).groupby(df.STATENAME).mean().round(3),
 'share_med_ded_demo': (g('MED_DED_DEMO')==1).groupby(df.STATENAME).mean().round(3),
 'share_special_calc(MFIP/SSI-CAP/demo)': df.groupby('STATENAME').special_calc.mean().round(3),
 'formula_reproduces_FSBEN_within_1': df.groupby('STATENAME').formula_ok.mean().round(3),
 'formula_ok_excl_special': df[~df.special_calc].groupby('STATENAME').formula_ok.mean().round(3),
})
# compare to Tech Doc F.7 simple rows
f7 = open(sys.argv[2],encoding='utf-8').read()
f7 = f7[f7.find('Table F.7'):]
ref={}
for m in re.finditer(r'^([A-Z][A-Za-z ]+?)\s+([\d,]+)\s+([\d,]+)\s+(\d+)\b', f7, flags=re.M):
    ref[m.group(1).strip()]=int(m.group(2).replace(',',''))
t['HCSUA_TechDoc_F7']=pd.Series(ref)
t['HCSUA_match']=np.where(t.HCSUA_TechDoc_F7.isna(),'n/a (multi-row or not parsed)', np.where(t.HCSUA_mode==t.HCSUA_TechDoc_F7,'yes','NO'))
t.to_csv(sys.argv[3])
print(t.HCSUA_match.value_counts().to_dict())
print(t[t.HCSUA_match=='NO'][['HCSUA_mode','HCSUA_TechDoc_F7']])
print("overall formula ok excl special:", round(df[~df.special_calc].formula_ok.mean(),4), " n special:", int(df.special_calc.sum()))
print(t.loc[['Michigan','California','Texas']].T.to_string())
