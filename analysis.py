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

logd[logd == 0] = 1
logd[(logd == 2) & (data.Medium != 'Air+2%')] = 1
#generate plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#colors dict
clrs = {'0.20%':'c','0.60%':'g','2%':'red','4%':'blue','Air+2%':'m'}
mindev = int(np.floor(data.deviation.min()))
maxdev = int(np.ceil(data.deviation.max()))
step = int((maxdev-mindev)/10)
bins = np.array(range(mindev,maxdev,step))
for ld in sorted(pd.unique(logd)):
    width = 0
    tmp = data[logd == ld]
    for m in pd.unique(tmp.Medium):
        tmp1 = tmp[tmp.Medium == m]

        #print tmp1.head(10)
        xy = np.histogram(tmp1.deviation.values,bins)
        ys = xy[0]
        xs = xy[1]+width
        ax.bar(xs[:-1]+step, ys, zs=[ld]*len(xs), zdir = 'y', color=[clrs[m]]*len(xs),
               width = np.ceil((xs[1:]-xs[:-1])[0]/len(pd.unique(tmp.Medium))))
        #print ld, clrs[m], (xs[1:]-xs[:-1])[0]/len(pd.unique(tmp.Medium))
        width += (xs[1:]-xs[:-1])[0]/len(pd.unique(tmp.Medium))
ax.set_xlabel('Deviation (%)')
ax.set_ylabel('log(gap size)')
ax.set_zlabel('count')
plt.show()
