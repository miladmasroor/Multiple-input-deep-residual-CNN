import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR

def R2 (true,predicted):
    import numpy as np
    R2 = np.mean((predicted-predicted.mean())*(true-true.mean()))/(np.std(predicted)*np.std(true))
    return R2

df1 = pd.read_excel('D:\\Hassan\\Work\\milad\\perm prediction\\data 3\\preds.xlsx')
df2 = pd.read_excel('D:\\Hassan\\Work\\milad\\perm prediction\\data 3\\blind_well_2.xlsx')

s1d = df1['pred_SI1D'].values.reshape(-1,1)
s2d = df1['pred_SI2D'].values.reshape(-1,1)
mi = df1['pred_MI'].values.reshape(-1,1)

t1 = df1['target'].values.reshape(-1,1)

#%%
res = t1-s1d
fig,ax = plt.subplots(3,1)

ax[0].plot(s1d,res,'o',alpha= 0.5, markeredgecolor='black', markerfacecolor = 'blue')
ax[0].plot([0,1],[0,0],'--', color = 'gray')
ax[0].set_ylabel('Residual Value', fontweight = 'bold')
ax[0].set_title('(a)',fontweight = 'bold')
ax[0].set_ylim([-.5,.5])

#%%
res = t1-s2d

ax[1].plot(s2d,res,'o',alpha= 0.5, markeredgecolor='black', markerfacecolor = 'blue')
ax[1].plot([0,1],[0,0],'--', color = 'gray')
ax[1].set_ylabel('Residual Value', fontweight = 'bold')
ax[1].set_title('(b)',fontweight = 'bold')
ax[1].set_ylim([-.5,.5])

#%%
res = t1-mi

ax[2].plot(mi,res,'o',alpha= 0.5, markeredgecolor='black', markerfacecolor = 'blue')
ax[2].plot([0,1],[0,0],'--', color = 'gray')
ax[2].set_xlabel('Predicted Value', fontweight = 'bold')
ax[2].set_ylabel('Residual Value', fontweight = 'bold')
ax[2].set_title('(c)',fontweight = 'bold')
ax[2].set_ylim([-.5,.5])