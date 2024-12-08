import pandas as pd 
import statsmodels.api as sm 
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))

df = pd.read_csv('states.csv')

#independent vars
x = sm.add_constant(df[['metro_res', 'white', 'hs_grad', 'female_house']])
#dependent var
y = df['poverty']

#teach the model
model = sm.OLS(y, x).fit()

fig.suptitle(model.summary())
print(model.summary())
plt.show()