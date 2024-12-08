import numpy as np 
import pandas as pd 
import scipy.stats as stats 
import statsmodels.api as sm 
import seaborn as sns
import matplotlib.pyplot as plt 

def calc_ols_statmodels(x, y):
    x_for_ols = sm.add_constant(x)
    model = sm.OLS(y, x_for_ols)
    results = model.fit()
    print('statsmodels:', results.summary())

    # regression formula
    if results.params.iloc[1] > 0:
        sign = "+"
    else:
        sign = "-"
    formula = f"y = {results.params.iloc[0]:.2f} {sign} {np.abs(results.params.iloc[1]):.2f} * x"
    print("-"*80)
    print("OLS Formula: ", formula)

    # Graphics
    fig = plt.figure(figsize=(16, 9), constrained_layout = True)
    gs = fig.add_gridspec(ncols = 3, nrows = 2)
    ax_main = fig.add_subplot(gs[0,:])
    ax_resid = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1,1])
    ax_qqplot = fig.add_subplot(gs[1,2])

    #Scatterplot
    sns.scatterplot(x=x, y=y, ax=ax_main, label='fact')
    ax_main.plot(x, results.predict(), color='red', alpha=0.5, label=formula)
    ax_main.set_title(f"Regression scatterplot. R2={results.rsquared:.2f}", fontsize=14)
    ax_main.set(xlabel='High School Graduation', ylabel='Poverty')
    ax_main.legend()

    #Residuals
    sns.scatterplot(x=x, y=results.resid, ax=ax_resid, label='residuals')
    ax_resid.hlines(0, x.min(), x.max(), linestyle='--', colors='red', alpha=0.5)
    ax_resid.set_title(f'Residuals vs Fitted Values', fontsize=14)
    ax_resid.set(xlabel='High School Graduation', ylabel='Residuals')

    #Hist
    sns.histplot(results.resid, ax=ax_hist, label='Distribution\n of Residuals')
    ax_hist.set_title('Histogram of Residuals', fontsize=14)
    ax_hist.legend()

    #QQplot
    # sm.qqplot(results.resid, ax=ax_qqplot)
    stats.probplot(results.resid, dist='norm', plot=ax_qqplot)
    ax_qqplot.set_title('Normal QQPlot of Residuals', fontsize=14)
    ax_qqplot.legend()




df = pd.read_csv('states.csv')
calc_ols_statmodels(df['hs_grad'], df['poverty'])
plt.show()