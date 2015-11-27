import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#import data
filename1 = '/Users/cgirabawe/Documents/OneDrive/DoD_Backup/BackUp_Data/allresults4f_tmp.csv'
df = pd.read_csv(filename1)
filename2 = '/Users/cgirabawe/Documents/OneDrive/DoD_Backup/BackUp_Data/datatbl.csv'
data = pd.read_csv(filename2)
gaps = df[['n10','n1f','n20','n2f']].min(axis = 1)
gaps = pd.DataFrame(gaps, columns = ['d'])
cols = data.columns
data = [data.c0,data.cf,data.v0,data.vf,data.expectedvf,data.expectedcf,data.deviation,gaps.d,df.Medium]
data = pd.DataFrame(data).T
logd = pd.Series(np.round(map(lambda x:np.log10(x+0.1), data.d)))


#generate plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax0 = fig.add_subplot(111, projection='3d')

#colors dict
clrs = {'0.20%':'m','0.60%':'c','2%':'red','4%':'blue','Air+2%':'y'}

for ld in sorted(pd.unique(logd))[0:1]:
    width = 0
    tmp = data[logd == ld]
    for m in pd.unique(tmp.Medium):
        tmp1 = tmp[tmp.Medium == m]
        #print clrs[m], width
        #print tmp1.head(10)
        xy = ax0.hist(tmp1.deviation.values)
        ys = xy[0]
        xs = xy[1]
        ax.bar(xs[:-1]+width, ys, zs=[ld]*len(xs), zdir = 'y', color=[clrs[m]]*len(xs))
        width += (xs[1:]-xs[:-1])[0]
ax.set_xlabel('Deviation (%)')
ax.set_ylabel('d (um)')
ax.set_zlabel('count')
plt.show()
