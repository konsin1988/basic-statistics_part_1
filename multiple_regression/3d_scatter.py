import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv('states.csv')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=df['white'], ys=df['poverty'], zs=df['hs_grad'])
plt.show()

