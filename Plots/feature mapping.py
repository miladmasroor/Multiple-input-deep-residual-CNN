import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as sg
import seaborn as sn
#%% A
df = pd.read_excel("D:\\Hassan\\Work\\milad\\perm prediction\\img\\data2.xlsx")
orig_df = df.copy()
tops = pd.read_excel("D:\\Hassan\\Work\\milad\\perm prediction\\img\\2.xlsx").values[:,1]


target = sg(df.pop('log(NMR_PERM)').values,31,3)
index = df.pop('index')
depth = df.pop('depth').values
features = df
del df

features['CGR'] = sg(features['CGR'].values,31,3)
features['NPHI'] = sg(features['NPHI'].values,31,3)
features['PEF'] = sg(features['PEF'].values,31,3)
features['RESD'] = sg(features['RESD'].values,31,3)
features['RESX'] = sg(features['RESX'].values,31,3)
features['RHOB'] = sg(features['RHOB'].values,31,3)
features['SGR'] = sg(features['SGR'].values,31,3)

Faetures = features.values.reshape(features.shape[0],features.shape[1],1)
Target = target.reshape(-1,1)

feat_num = Faetures.reshape(Faetures.shape[0],Faetures.shape[1],1)

feat_img = []
for i in feat_num:
    A,B = np.meshgrid(i,i)
    Z = A*B
    Z = Z.reshape(Z.shape[0],Z.shape[1],1)
    feat_img.append(Z)
feat_img = np.array(feat_img)


tps = []
c = 0
c_end = len(tops)-1
for i in depth:
    if c <= c_end:
        if i<= tops[c]:
            tps.append([c])
        else:
            c+=1
            tps.append([c])
    else:
        tps.append([c])
        
tps = np.array(tps)
features['Tops'] = tps

fig, ax = plt.subplots(1, 9)

ax[0].plot(features['CGR'],depth,'r')
ax[0].xaxis.tick_top()
ax[0].set_title('Normalized CGR',fontsize=10)
ax[0].set_ylabel('Depth(m)')
ax[0].grid()
ax[0].invert_yaxis()
ax[0].set_ylim(depth[-1],depth[0])
ax[0].set_xlim(0,1)
ax[0].set_xticks([0,0.25,0.5,.75,1])
ax[0].set_xticklabels(['0','','0.5','','1'])
'''
ax[0].axhline(y = 2686, color = 'r', linestyle = ':')
ax[0].axhline(y = 2707, color = 'r', linestyle = ':')
ax[0].axhline(y = 2716, color = 'r', linestyle = ':')
ax[0].axhline(y = 2748, color = 'r', linestyle = ':')
ax[0].axhline(y = 2840, color = 'r', linestyle = ':')
ax[0].axhline(y = 2857, color = 'r', linestyle = ':')
ax[0].axhline(y = 2911, color = 'r', linestyle = ':')
ax[0].axhline(y = 2948, color = 'r', linestyle = ':')
ax[0].axhline(y = 2978, color = 'r', linestyle = ':')
ax[0].axhline(y = 3008, color = 'r', linestyle = ':')
ax[0].axhline(y = 3096, color = 'r', linestyle = ':')
'''
b = ax[0].get_yticks()
bb = []
for i in b:
    bb.append('')


ax[1].plot(features['NPHI'],depth,'g')
ax[1].xaxis.tick_top()
ax[1].set_title('Normalized NPHI',fontsize=10)
ax[1].grid()
ax[1].invert_yaxis()
ax[1].set_ylim(depth[-1],depth[0])
ax[1].set_xlim(0,1)
ax[1].set_xticks([0,0.25,0.5,.75,1])
ax[1].set_xticklabels(['0','','0.5','','1'])
ax[1].set_yticklabels(bb)
'''
ax[1].axhline(y = 2686, color = 'g', linestyle = ':')
ax[1].axhline(y = 2707, color = 'g', linestyle = ':')
ax[1].axhline(y = 2716, color = 'g', linestyle = ':')
ax[1].axhline(y = 2748, color = 'g', linestyle = ':')
ax[1].axhline(y = 2840, color = 'g', linestyle = ':')
ax[1].axhline(y = 2857, color = 'g', linestyle = ':')
ax[1].axhline(y = 2911, color = 'g', linestyle = ':')
ax[1].axhline(y = 2948, color = 'g', linestyle = ':')
ax[1].axhline(y = 2978, color = 'g', linestyle = ':')
ax[1].axhline(y = 3008, color = 'g', linestyle = ':')
ax[1].axhline(y = 3096, color = 'g', linestyle = ':')
'''

ax[2].plot(features['PEF'],depth,'k')
ax[2].xaxis.tick_top()
ax[2].set_title('Normalized PEF',fontsize=10)
ax[2].grid()
ax[2].invert_yaxis()
ax[2].set_ylim(depth[-1],depth[0])
ax[2].set_xlim(0,1)
ax[2].set_xticks([0,0.25,0.5,.75,1])
ax[2].set_xticklabels(['0','','0.5','','1'])
ax[2].set_yticklabels(bb)
'''
ax[2].axhline(y = 2686, color = 'k', linestyle = ':')
ax[2].axhline(y = 2707, color = 'k', linestyle = ':')
ax[2].axhline(y = 2716, color = 'k', linestyle = ':')
ax[2].axhline(y = 2748, color = 'k', linestyle = ':')
ax[2].axhline(y = 2840, color = 'k', linestyle = ':')
ax[2].axhline(y = 2857, color = 'k', linestyle = ':')
ax[2].axhline(y = 2911, color = 'k', linestyle = ':')
ax[2].axhline(y = 2948, color = 'k', linestyle = ':')
ax[2].axhline(y = 2978, color = 'k', linestyle = ':')
ax[2].axhline(y = 3008, color = 'k', linestyle = ':')
ax[2].axhline(y = 3096, color = 'k', linestyle = ':')
ax[2].axhline(y = 2914, color = 'k', linestyle = ':')
'''


ax[3].plot(features['RESD'],depth,'y')
ax[3].xaxis.tick_top()
ax[3].set_title('Normalized LLD',fontsize=10)
ax[3].grid()
ax[3].invert_yaxis()
ax[3].set_ylim(depth[-1],depth[0])
ax[3].set_xlim(0,1)
ax[3].set_xticks([0,0.25,0.5,.75,1])
ax[3].set_xticklabels(['0','','0.5','','1'])
ax[3].set_yticklabels(bb)
'''
ax[3].axhline(y = 2686, color = 'y', linestyle = ':')
ax[3].axhline(y = 2707, color = 'y', linestyle = ':')
ax[3].axhline(y = 2716, color = 'y', linestyle = ':')
ax[3].axhline(y = 2748, color = 'y', linestyle = ':')
ax[3].axhline(y = 2840, color = 'y', linestyle = ':')
ax[3].axhline(y = 2857, color = 'y', linestyle = ':')
ax[3].axhline(y = 2911, color = 'y', linestyle = ':')
ax[3].axhline(y = 2948, color = 'y', linestyle = ':')
ax[3].axhline(y = 2978, color = 'y', linestyle = ':')
ax[3].axhline(y = 3008, color = 'y', linestyle = ':')
ax[3].axhline(y = 3096, color = 'y', linestyle = ':')
'''


ax[4].plot(features['RESX'],depth,'orange')
ax[4].xaxis.tick_top()
ax[4].set_title('Normalized LLS',fontsize=10)
ax[4].grid()
ax[4].invert_yaxis()
ax[4].set_ylim(depth[-1],depth[0])
ax[4].set_xlim(0,1)
ax[4].set_xticks([0,0.25,0.5,.75,1])
ax[4].set_xticklabels(['0','','0.5','','1'])
ax[4].set_yticklabels(bb)
'''
ax[4].axhline(y = 2686, color = 'orange', linestyle = ':')
ax[4].axhline(y = 2707, color = 'orange', linestyle = ':')
ax[4].axhline(y = 2716, color = 'orange', linestyle = ':')
ax[4].axhline(y = 2748, color = 'orange', linestyle = ':')
ax[4].axhline(y = 2840, color = 'orange', linestyle = ':')
ax[4].axhline(y = 2857, color = 'orange', linestyle = ':')
ax[4].axhline(y = 2911, color = 'orange', linestyle = ':')
ax[4].axhline(y = 2948, color = 'orange', linestyle = ':')
ax[4].axhline(y = 2978, color = 'orange', linestyle = ':')
ax[4].axhline(y = 3008, color = 'orange', linestyle = ':')
ax[4].axhline(y = 3096, color = 'orange', linestyle = ':')
'''

ax[5].plot(features['RHOB'],depth,'purple')
ax[5].xaxis.tick_top()
ax[5].set_title('Normalized RHOB',fontsize=10)
ax[5].grid()
ax[5].invert_yaxis()
ax[5].set_ylim(depth[-1],depth[0])
ax[5].set_xlim(0,1)
ax[5].set_xticks([0,0.25,0.5,.75,1])
ax[5].set_xticklabels(['0','','0.5','','1'])
ax[5].set_yticklabels(bb)
'''
ax[5].axhline(y = 2686, color = 'purple', linestyle = ':')
ax[5].axhline(y = 2707, color = 'purple', linestyle = ':')
ax[5].axhline(y = 2716, color = 'purple', linestyle = ':')
ax[5].axhline(y = 2748, color = 'purple', linestyle = ':')
ax[5].axhline(y = 2840, color = 'purple', linestyle = ':')
ax[5].axhline(y = 2857, color = 'purple', linestyle = ':')
ax[5].axhline(y = 2911, color = 'purple', linestyle = ':')
ax[5].axhline(y = 2948, color = 'purple', linestyle = ':')
ax[5].axhline(y = 2978, color = 'purple', linestyle = ':')
ax[5].axhline(y = 3008, color = 'purple', linestyle = ':')
ax[5].axhline(y = 3096, color = 'purple', linestyle = ':')
'''

ax[6].plot(features['SGR'],depth,'b')
ax[6].xaxis.tick_top()
ax[6].set_title('Normalized SGR',fontsize=10)
ax[6].grid()
ax[6].invert_yaxis()
ax[6].set_ylim(depth[-1],depth[0])
ax[6].set_xlim(0,1)
ax[6].set_xticks([0,0.25,0.5,.75,1])
ax[6].set_xticklabels(['0','','0.5','','1'])
ax[6].set_yticklabels(bb)
'''
ax[6].axhline(y = 2686, color = 'b', linestyle = ':')
ax[6].axhline(y = 2707, color = 'b', linestyle = ':')
ax[6].axhline(y = 2716, color = 'b', linestyle = ':')
ax[6].axhline(y = 2748, color = 'b', linestyle = ':')
ax[6].axhline(y = 2840, color = 'b', linestyle = ':')
ax[6].axhline(y = 2857, color = 'b', linestyle = ':')
ax[6].axhline(y = 2911, color = 'b', linestyle = ':')
ax[6].axhline(y = 2948, color = 'b', linestyle = ':')
ax[6].axhline(y = 2978, color = 'b', linestyle = ':')
ax[6].axhline(y = 3008, color = 'b', linestyle = ':')
ax[6].axhline(y = 3096, color = 'b', linestyle = ':')
'''


a = feat_num[0].reshape(1,7)
b = feat_num[1].reshape(1,7)
c = np.r_[a,b]
for i in range (2,len(feat_img)):
    c = np.r_[c,feat_num[i].reshape(1,7)]

ax_7 = ax[7].imshow(c, aspect='auto'
             , interpolation='none'
             #, vmin=10, vmax=60
             ,cmap='viridis')
ax[7].xaxis.tick_top()
ax[7].set_title('Graphical\nFeature Log',fontsize=10)
#ax[7].set_xticks([i for i in depth])
ax[7].axis('off')
cbar = plt.colorbar(ax_7, ax = ax[7])
'''
plt.figure()
sn.heatmap(c ,cmap='viridis')
'''

ax[8].imshow(tps, aspect='auto'
             , interpolation='none'
             #, vmin=10, vmax=60
             ,cmap=plt.cm.Spectral)
ax[8].set_title('Subzones\n',fontsize=10)
ax[8].axis('off')

c = 0
c_list = []
for i in tps:
    if c>0:
        if tps[c]!=tps[c-1]:
            c_list.append(c)
    c+=1

for i in c_list[0:]:
    ax[8].axhline(y = i, color = 'k', linestyle = '-')


#%%%%%%%%%
fig, ax = plt.subplots(1, 3)

ax[0].imshow(tps, aspect='auto'
             , interpolation='none'
             #, vmin=10, vmax=60
             ,cmap=plt.cm.RdBu_r)
ax[0].set_title('Zone\n',fontsize=10)
ax[0].set_xticks([])
ax[0].set_ylabel('Depth(m)')

a = feat_num[0].reshape(1,7)
b = feat_num[1].reshape(1,7)
c = np.r_[a,b]
for i in range (2,len(feat_img)):
    c = np.r_[c,feat_num[i].reshape(1,7)]

ax_7 = ax[1].imshow(c, aspect='auto'
             , interpolation='none'
             ,cmap='viridis')
ax[1].xaxis.tick_top()
ax[1].set_title('Graphical\nFeature Log',fontsize=10)
ax[1].axis('off')

a = feat_img[0].reshape(7,7)
b = feat_img[1].reshape(7,7)
c = np.r_[a,b]
for i in range (2,len(feat_img)):
    c = np.r_[c,feat_img[i].reshape(7,7)]

ax_7 = ax[2].imshow(c, aspect='auto'
             , interpolation='none'
             ,cmap='viridis')
ax[2].xaxis.tick_top()
ax[2].set_title('Engineered Graphical\nFeature Log',fontsize=10)
ax[2].axis('off')
cbar = plt.colorbar(ax_7, ax = ax[2])