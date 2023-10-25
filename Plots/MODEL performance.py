import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from scipy.io import loadmat
import scipy.stats as st

def MAPE(true,predicted):
    n = true.shape[0]
    mape = 1/n * sum(abs(true-predicted)/true)
    return mape[0]

def R2 (true,predicted):
    import numpy as np
    R2 = np.mean((predicted-predicted.mean())*(true-true.mean()))/(np.std(predicted)*np.std(true))
    return R2
def R(true,predicted):
    R = (np.sum((true-true.mean())*(predicted-predicted.mean())))/(np.sqrt((np.sum((true-true.mean())**2))*np.sum((true-true.mean())**2)))
    return R

_max_ = 2.77005
_min_ = -5

''' blind well
_max_ = 3.3872
_min_ = -4
'''


fig,axis = plt.subplots(3,2)
#%% res1d

y_pred_ = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\res1d\\matlab_matrix_tr_pred_target_res1d.mat")['pred'].reshape(-1,1)
y_ = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\res1d\\matlab_matrix_tr_true_target_res1d.mat")['true'].reshape(-1,1)

y = y_*(_max_-_min_)+_min_
y_pred = y_pred_*(_max_-_min_)+_min_

diff = (y-y_pred)
mu = diff.mean().astype('float16')
alpha = diff.std().astype('float16')

mse = MSE(y,y_pred).astype('float16')
rmse = (mse**0.5).astype('float16')
r2 = R2(y_pred,y).astype('float16')
mae = MAE(y,y_pred).astype('float16')
mape = MAPE(y,y_pred).astype('float16')

plt.figure()
a = plt.hist(diff,bins=50,edgecolor='k',color='purple')[0]
mn, mx = plt.xlim()

#2
axis[0][0].plot(y_pred,y,'.k')
axis[0][0].plot([-5,3],[-5,3],'r')
#axis[0][0].set_xlabel('Predicted Normalized log(K)', fontsize=15)
axis[0][0].set_ylabel('Real log(K)', fontsize=15)
axis[0][0].set_title('(a)',fontweight='bold', fontsize=15)
axis[0][0].text(-5, 0, 'R='+str(r2),bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},
        color='k', fontsize=10,fontweight='bold')
axis[0][0].set_yticks([-5,-3,-1,1,3])
axis[0][0].xaxis.set_tick_params(labelsize=15)
axis[0][0].yaxis.set_tick_params(labelsize=15)

#3
axis[0][1].hist(diff,bins=50,edgecolor='k',color='purple')
kde_xs = np.linspace(mn, mx,300)
kde = st.gaussian_kde(diff.reshape(diff.shape[0],))
n_data = y_pred.shape[0]
width = (mx-mn)/(300-1)
axis[0][1].plot(kde_xs, kde.pdf(kde_xs)*width*n_data*6)
#axis[0][1].set_xlabel('Error', fontsize=15)
axis[0][1].set_xlim([-5,3])
axis[0][1].set_ylabel('Frequency', fontsize=15)
axis[0][1].set_ylim([0,750])
axis[0][1].set_title('(b)' ,fontweight='bold', fontsize=15)
axis[0][1].text(-4, 200, 'Mean=' +str(mu)+'\nStd='+str(alpha) ,bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},
        color='k', fontsize=10,fontweight='bold')
axis[0][1].xaxis.set_tick_params(labelsize=15)
axis[0][1].yaxis.set_tick_params(labelsize=15)

#%% res2d

y_pred_ = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\res2d\\matlab_matrix_tr_pred_target_res2d.mat")['pred'].reshape(-1,1)
y_ = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\res2d\\matlab_matrix_tr_true_target_res2d.mat")['true'].reshape(-1,1)

y = y_*(_max_-_min_)+_min_
y_pred = y_pred_*(_max_-_min_)+_min_

diff = (y-y_pred)
mu = diff.mean().astype('float16')
alpha = diff.std().astype('float16')

mse = MSE(y,y_pred).astype('float16')
rmse = (mse**0.5).astype('float16')
r2 = R2(y_pred,y).astype('float16')
mae = MAE(y,y_pred).astype('float16')
mape = MAPE(y,y_pred).astype('float16')

plt.figure()
a = plt.hist(diff,bins=50,edgecolor='k',color='purple')[0]
mn, mx = plt.xlim()

#2
axis[1][0].plot(y_pred,y,'.k')
axis[1][0].plot([-5,3],[-5,3],'r')
#axis[1][0].set_xlabel('Predicted Normalized log(K)', fontsize=15)
axis[1][0].set_ylabel('Real log(K)', fontsize=15)
axis[1][0].set_title('(c)',fontweight='bold', fontsize=15)
axis[1][0].text(-5, 0, 'R='+str(r2),bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},
        color='k', fontsize=10,fontweight='bold')
axis[1][0].set_yticks([-5,-3,-1,1,3])
axis[1][0].xaxis.set_tick_params(labelsize=15)
axis[1][0].yaxis.set_tick_params(labelsize=15)

#3
axis[1][1].hist(diff,bins=50,edgecolor='k',color='purple')
kde_xs = np.linspace(mn, mx,300)
kde = st.gaussian_kde(diff.reshape(diff.shape[0],))
n_data = y_pred.shape[0]
width = (mx-mn)/(300-1)
axis[1][1].plot(kde_xs, kde.pdf(kde_xs)*width*n_data*6)
#axis[0][1].set_xlabel('Error', fontsize=15)
axis[1][1].set_xlim([-5,3])
axis[1][1].set_ylabel('Frequency', fontsize=15)
axis[1][1].set_ylim([0,750])
axis[1][1].set_title('(d)' ,fontweight='bold', fontsize=15)
axis[1][1].text(-4, 200, 'Mean=' +str(mu)+'\nStd='+str(alpha) ,bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},
        color='k', fontsize=10,fontweight='bold')
axis[1][1].xaxis.set_tick_params(labelsize=15)
axis[1][1].yaxis.set_tick_params(labelsize=15)

#%% multi

y_pred_ = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\multi\\matlab_matrix_tr_pred_target_multi.mat")['pred'].reshape(-1,1)
y_ = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\multi\\matlab_matrix_tr_true_target_multi.mat")['true'].reshape(-1,1)

y = y_*(_max_-_min_)+_min_
y_pred = y_pred_*(_max_-_min_)+_min_

diff = (y-y_pred)
mu = diff.mean().astype('float16')
alpha = diff.std().astype('float16')

mse = MSE(y,y_pred).astype('float16')
rmse = (mse**0.5).astype('float16')
r2 = R2(y_pred,y).astype('float16')
mae = MAE(y,y_pred).astype('float16')
mape = MAPE(y,y_pred).astype('float16')

y = y_*(_max_-_min_)+_min_
y_pred = y_pred_*(_max_-_min_)+_min_

plt.figure()
a = plt.hist(diff,bins=50,edgecolor='k',color='purple')[0]
mn, mx = plt.xlim()

#2
axis[2][0].plot(y_pred,y,'.k')
axis[2][0].plot([-5,3],[-5,3],'r')
#axis[1][0].set_xlabel('Predicted Normalized log(K)', fontsize=15)
axis[2][0].set_ylabel('Real log(K)', fontsize=15)
axis[2][0].set_xlabel('Predicted log(K)', fontsize=15)

axis[2][0].set_title('(e)',fontweight='bold', fontsize=15)
axis[2][0].text(-5, 0, 'R='+str(r2),bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},
        color='k', fontsize=10,fontweight='bold')
axis[2][0].set_yticks([-5,-3,-1,1,3])
axis[2][0].xaxis.set_tick_params(labelsize=15)
axis[2][0].yaxis.set_tick_params(labelsize=15)

#3
axis[2][1].hist(diff,bins=50,edgecolor='k',color='purple')
kde_xs = np.linspace(mn, mx,300)
kde = st.gaussian_kde(diff.reshape(diff.shape[0],))
n_data = y_pred.shape[0]
width = (mx-mn)/(300-1)
axis[2][1].plot(kde_xs, kde.pdf(kde_xs)*width*n_data*6)
#axis[0][1].set_xlabel('Error', fontsize=15)
axis[2][1].set_xlim([-5,3])
axis[2][1].set_ylabel('Frequency', fontsize=15)
axis[2][1].set_xlabel('Error', fontsize=15)
axis[2][1].set_ylim([0,750])
axis[2][1].set_title('(f)' ,fontweight='bold', fontsize=15)
axis[2][1].text(-4, 200, 'Mean=' +str(mu)+'\nStd='+str(alpha) ,bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},
        color='k', fontsize=10,fontweight='bold')
axis[2][1].xaxis.set_tick_params(labelsize=15)
axis[2][1].yaxis.set_tick_params(labelsize=15)