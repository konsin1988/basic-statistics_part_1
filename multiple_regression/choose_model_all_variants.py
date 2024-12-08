import itertools
import pandas as pd 
import statsmodels.api as sm 

df = pd.read_csv('states.csv').drop('state', axis=1)
res = []
feats = df.columns[df.columns != 'poverty']
d = {}

for i in range(1, len(feats) + 1):
    res.extend([list(x) for x in itertools.combinations(feats, i)])

y = df['poverty']
for i in res:
    x = df[i]
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    d[' '.join(i)] = round(float(model.rsquared_adj), 3)


res = [ (x[1], x[0]) for x in d.items()]
res.sort(reverse=True)
[print(f'{x[0]} - {x[1]}') for x in res]
    