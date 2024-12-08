import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('states.csv')
plt.rcParams['figure.figsize'] = (11,7)


# simple with text
# axes = pd.plotting.scatter_matrix(df, diagonal='kde', grid=True)
# corr = df.corr(numeric_only=True).values
# for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
#     axes[i, j].annotate('%.3f' %corr[i, j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')

def update_plot(x, y, **kws):
    model = LinearRegression()
    model.fit(pd.DataFrame(x), pd.DataFrame(y))
    plt.axline([0, model.intercept_[0]], slope=model.coef_[0][0], color='crimson', linestyle='--')
    plt.annotate(f'r = {"%.3f" %corr[x.name][y.name]}', xy=(0, 35), xytext=(0.8, 0.9), textcoords='axes fraction', ha='center', va='center')
    

#seaborn 
corr = df.corr(numeric_only=True).values
corr = pd.DataFrame(corr, columns=df.columns[1:], index=df.columns[1:])
g = sns.PairGrid(df, diag_sharey=False)
g.map_offdiag(plt.scatter, edgecolor='w')
g.map_offdiag(update_plot)
g.map_diag(sns.kdeplot, fill=True)

#seaborn var.II



plt.show()

