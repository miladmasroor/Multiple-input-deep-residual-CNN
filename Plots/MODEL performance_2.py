import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from scipy.io import loadmat

def MAPE(true,predicted):
    mape = np.mean(np.abs((true - predicted) / true))
    return mape

def R2 (true,predicted):
    import numpy as np
    R2 = np.mean((predicted-predicted.mean())*(true-true.mean()))/(np.std(predicted)*np.std(true))
    return R2
def R(true,predicted):
    R = (np.sum((true-true.mean())*(predicted-predicted.mean())))/(np.sqrt((np.sum((true-true.mean())**2))*np.sum((true-true.mean())**2)))
    return R


#%% train and test wells
_max_ = 2.77005
_min_ = -5

# train
y = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\res1d\\matlab_matrix_ts_true_target_res1d.mat")['true'].reshape(-1,1)
y = y*(_max_-_min_)+_min_
y_list = []
c = 0
for i in y:
    if c!=15:
        c+=1
    else:
        y_list.append(i)
        c = 0
y = np.array(y_list).reshape(-1,1)

y_pred_1d = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\res1d\\matlab_matrix_ts_pred_target_res1d.mat")['pred'].reshape(-1,1)
y_pred_1d = y_pred_1d*(_max_-_min_)+_min_
y_list = []
c = 0
for i in y_pred_1d:
    if c!=15:
        c+=1
    else:
        y_list.append(i)
        c = 0
y_pred_1d = np.array(y_list).reshape(-1,1)

y_pred_2d = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\res2d\\matlab_matrix_ts_pred_target_res2d.mat")['pred'].reshape(-1,1)
y_pred_2d = y_pred_2d*(_max_-_min_)+_min_
y_list = []
c = 0
for i in y_pred_2d:
    if c!=15:
        c+=1
    else:
        y_list.append(i)
        c = 0
y_pred_2d = np.array(y_list).reshape(-1,1)

y_pred_multi = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\multi\\matlab_matrix_ts_pred_target_multi.mat")['pred'].reshape(-1,1)
y_pred_multi = y_pred_multi*(_max_-_min_)+_min_
y_list = []
c = 0
for i in y_pred_multi:
    if c!=15:
        c+=1
    else:
        y_list.append(i)
        c = 0
y_pred_multi = np.array(y_list).reshape(-1,1)

depth = np.linspace(0,y.shape[0]-1,y.shape[0])*15

# test

#%% blind well
_max_ = 3.3872
_min_ = -4

y = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\res1d\\matlab_matrix_b_true_target_res1d.mat")['true'].reshape(-1,1)
y = y*(_max_-_min_)+_min_

y_pred_1d = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\res1d\\matlab_matrix_b_pred_target_res1d.mat")['pred'].reshape(-1,1)
y_pred_1d = y_pred_1d*(_max_-_min_)+_min_

y_pred_2d = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\res2d\\matlab_matrix_b_pred_target_res2d.mat")['pred'].reshape(-1,1)
y_pred_2d = y_pred_2d*(_max_-_min_)+_min_

y_pred_multi = loadmat("D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\multi\\matlab_matrix_b_pred_target_multi.mat")['pred'].reshape(-1,1)
y_pred_multi = y_pred_multi*(_max_-_min_)+_min_

df = pd.read_excel("D:\\Hassan\\Work\\milad\\perm prediction\\img_b\\data3.xlsx")
depth = df['depth'].values.reshape(-1,1)
#%%
fig,axis = plt.subplots(3,1)

#%%1d
diff = (y-y_pred_1d)
mu = diff.mean().astype('float16')
alpha = diff.std().astype('float16')

mse = MSE(y,y_pred_1d).astype('float16')
rmse = (mse**0.5).astype('float16')
r2 = R2(y_pred_1d,y).astype('float16')
mae = MAE(y,y_pred_1d).astype('float16')
mape = MAPE(y,y_pred_1d).astype('float16')

axis[0].plot(depth,y,'k',label='Real')
axis[0].plot(depth,y_pred_1d,'r',label='Predicted')
axis[0].set_ylabel('log(K)'
                   #,fontweight='bold'
                   , fontsize=15)
axis[0].set_title('(a) '+'MSE=' +str(mse)+', MAE='+str(mae),
        color='k', fontsize=15,fontweight='bold')
axis[0].set_ylim([-7,5])
axis[0].set_yticks([-7,-5,-3,-1,1,3,5])
axis[0].xaxis.set_tick_params(labelsize=15)
axis[0].yaxis.set_tick_params(labelsize=15)
legend_properties = {'weight':'bold'}

axis2 = axis[0].twinx()
axis2.plot(depth,diff,label='Error',color='gray')
axis2.set_ylim([-2,10])
axis2.set_ylabel('Error', fontsize=15)

axis[0].legend(loc="upper left",prop=legend_properties)
axis2.legend(loc="upper right",prop=legend_properties)
#%%2d
diff = (y-y_pred_2d)
mu = diff.mean().astype('float16')
alpha = diff.std().astype('float16')

mse = MSE(y,y_pred_2d).astype('float16')
rmse = (mse**0.5).astype('float16')
r2 = R2(y_pred_2d,y).astype('float16')
mae = MAE(y,y_pred_2d).astype('float16')
mape = MAPE(y,y_pred_2d).astype('float16')

axis[1].plot(depth,y,'k',label='Real')
axis[1].plot(depth,y_pred_2d,'r',label='Predicted')
axis[1].set_ylabel('log(K)'
                   #,fontweight='bold'
                   , fontsize=15)
axis[1].set_title('(b) '+'MSE=' +str(mse)+', MAE='+str(mae),
        color='k', fontsize=15,fontweight='bold')
axis[1].set_ylim([-7,5])
axis[1].set_yticks([-7,-5,-3,-1,1,3,5])
axis[1].xaxis.set_tick_params(labelsize=15)
axis[1].yaxis.set_tick_params(labelsize=15)
legend_properties = {'weight':'bold'}

axis2 = axis[1].twinx()
axis2.plot(depth,diff,label='Error',color='gray')
axis2.set_ylim([-2,10])
axis2.set_ylabel('Error', fontsize=15)

axis[1].legend(loc="upper left",prop=legend_properties)
axis2.legend(loc="upper right",prop=legend_properties)
#%%multi

diff = (y-y_pred_multi)
mu = diff.mean().astype('float16')
alpha = diff.std().astype('float16')

mse = MSE(y,y_pred_multi).astype('float16')
rmse = (mse**0.5).astype('float16')
r2 = R2(y_pred_multi,y).astype('float16')
mae = MAE(y,y_pred_multi).astype('float16')
mape = MAPE(y,y_pred_multi).astype('float16')

axis[2].plot(depth,y,'k',label='Real')
axis[2].plot(depth,y_pred_multi,'r',label='Predicted')
axis[2].set_ylabel('log(K)'
                   #,fontweight='bold'
                   , fontsize=15)
axis[2].set_title('(c) '+'MSE=' +str(mse)+', MAE='+str(mae),
        color='k', fontsize=15,fontweight='bold')
axis[2].set_ylim([-7,5])
axis[2].set_yticks([-7,-5,-3,-1,1,3,5])
axis[2].xaxis.set_tick_params(labelsize=15)
axis[2].yaxis.set_tick_params(labelsize=15)
legend_properties = {'weight':'bold'}

axis2 = axis[2].twinx()
axis2.plot(depth,diff,label='Error',color='gray')
axis2.set_ylim([-2,10])
axis2.set_ylabel('Error', fontsize=15)

axis[2].legend(loc="upper left",prop=legend_properties)
axis[2].set_xlabel('Sample',fontsize=15)
axis2.legend(loc="upper right",prop=legend_properties)