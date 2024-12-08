import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('states.csv', index_col = 0)

# Попарное выведение рультатов на графики
# sns.set_theme(style='white')
# g = sns.pairplot(data=df)
# plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
sns.regplot(
    x = 'hs_grad',
    y = "poverty",
    data = df,
    scatter_kws = {'color':'blue'},
    line_kws = {'color': 'red'}
)
sns.despine()
plt.show()