import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
import numpy as np 

df = pd.read_csv('states.csv')

def residual_is_positive(a, b, y):
    return model.predict(pd.DataFrame({'hs_grad': a, 'white': b}, index=[0])) < y

y = df['poverty']
X = df[['hs_grad', 'white']]
model = LinearRegression()
model.fit(X, y)
print(f'Intercept: {model.intercept_:.3f}')
print(f'Coefficient Exposure: {model.coef_}')

df['residual_is_positive'] = np.vectorize(residual_is_positive)(df['hs_grad'], df['white'], df['poverty'])

#Draw the 3D diagramm
plt.figure(figsize=(6,6))
ax = plt.axes(projection = '3d')
ax.scatter3D(X['hs_grad'], X['white'], y, s=50, c=df['residual_is_positive'], cmap=ListedColormap(['blue', 'green']))
ax.set_xlabel('Pecent of hs_grad')
ax.set_ylabel('Pecent of white citizen')
ax.set_zlabel('Poverty')
plt.title('Multiple regression', fontdict={'size': 18})

#Count the coordinates of net and draw the plane 
n = 10 
'''Переменная для создания сетки n*n.
В данному случае количество узлов сетки 100 избыточно, 
но в дальнейшем при использовании этой заготовки
для полиноминальной регрессии может пригодиться'''
x_m, y_m = np.meshgrid(np.linspace(*ax.get_xlim(), n), np.linspace(*ax.get_ylim(), n))
z_m = model.predict(np.concatenate([x_m.reshape(-1, 1), y_m.reshape(-1, 1)], axis=1)).reshape(n, n)
ax.plot_wireframe(x_m, y_m, z_m, alpha=0.5, color='red')
plt.show()