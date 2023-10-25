import matplotlib.pyplot as plt
import numpy as np

b_1d = np.array([0.6353,0.4038,0.5347,1.827,0.9316])
b_2d = np.array([0.4883,0.2384,0.41,1.59,0.9453])
b_multi = np.array([0.2306,0.0532,0.1707,0.805,0.985])

b2_1d = np.array([0.4846,0.2347,0.3545,1.406,0.8013])
b2_2d = np.array([0.529,0.2795,0.3457,1.601,0.8667])
b2_multi = np.array([0.3853,0.1484,0.3218,1.463,0.919])

tr_1d = np.array([0.4202,0.1766,0.305,1.22,0.9556])
tr_2d = np.array([0.4592,0.2109,.3362,1.062,.9526])
tr_multi = np.array([0.1948,.03796,.1448,.6787,.9917])

ts_1d = np.array([0.4595,.211,.3188,1.42,.9497])
ts_2d = np.array([0.4976,.2477,.3484,1.145,.9463])
ts_multi = np.array([0.2231,.04977,.1652,1.109,.9893])

fig,axis = plt.subplots(1,4)
legend_properties = {'weight':'bold'}
#%% R2
# Train
labels = [1,6,11]
axis[0].bar(labels,[tr_1d[-1],tr_2d[-1],tr_multi[-1]],label='Train Set')

# Test
labels = [2,7,12]
axis[0].bar(labels,[ts_1d[-1],ts_2d[-1],ts_multi[-1]],label='Test Set')

# Test
labels = [3,8,13]
axis[0].bar(labels,[b_1d[-1],b_2d[-1],b_multi[-1]],label='Blind Set')

axis[0].set_xticks([2,7,12])
axis[0].set_xticklabels(['SIRes 1D-CNN','SIRes 2D-CNN','MIRes CNN'],fontsize=15
                        ,rotation=45)
axis[0].set_ylim([0,1.2])
axis[0].yaxis.set_tick_params(labelsize=15)
axis[0].legend(loc='best',prop=legend_properties)
axis[0].set_title('(a)',fontsize=15,fontweight = 'bold')
axis[0].set_ylabel('R',rotation=90,fontsize=15)

#%% MSE
# Train
labels = [1,6,11]
axis[1].bar(labels,[tr_1d[1],tr_2d[1],tr_multi[1]],label='Train Set')

# Test
labels = [2,7,12]
axis[1].bar(labels,[ts_1d[1],ts_2d[1],ts_multi[1]],label='Test Set')

# Test
labels = [3,8,13]
axis[1].bar(labels,[b_1d[1],b_2d[1],b_multi[1]],label='Blind Set')

axis[1].set_xticks([2,7,12])
axis[1].set_xticklabels(['SIRes 1D-CNN','SIRes 2D-CNN','MIRes CNN'],fontsize=15
                        ,rotation=45)
#axis[1].set_ylim([0,1.2])
axis[1].yaxis.set_tick_params(labelsize=15)
axis[1].legend(loc='best',prop=legend_properties)
axis[1].set_title('(b)',fontsize=15,fontweight = 'bold')
axis[1].set_ylabel('MSE',rotation=90,fontsize=15)

#%% MAE
# Train
labels = [1,6,11]
axis[2].bar(labels,[tr_1d[2],tr_2d[2],tr_multi[2]],label='Train Set')

# Test
labels = [2,7,12]
axis[2].bar(labels,[ts_1d[2],ts_2d[2],ts_multi[2]],label='Test Set')

# Test
labels = [3,8,13]
axis[2].bar(labels,[b_1d[2],b_2d[2],b_multi[2]],label='Blind Set')

axis[2].set_xticks([2,7,12])
axis[2].set_xticklabels(['SIRes 1D-CNN','SIRes 2D-CNN','MIRes CNN'],fontsize=15
                       ,rotation=45)
#axis[1].set_ylim([0,1.2])
axis[2].yaxis.set_tick_params(labelsize=15)
axis[2].legend(loc='best',prop=legend_properties)
axis[2].set_title('(c)',fontsize=15,fontweight = 'bold')
axis[2].set_ylabel('MAE',rotation=90,fontsize=15)

#%% MAPE
# Train
labels = [1,6,11]
axis[3].bar(labels,[tr_1d[3],tr_2d[3],tr_multi[3]],label='Train Set')

# Test
labels = [2,7,12]
axis[3].bar(labels,[ts_1d[3],ts_2d[3],ts_multi[3]],label='Test Set')

# Test
labels = [3,8,13]
axis[3].bar(labels,[b_1d[3],b_2d[3],b_multi[3]],label='Blind Set')

axis[3].set_xticks([2,7,12])
axis[3].set_xticklabels(['SIRes 1D-CNN','SIRes 2D-CNN','MIRes CNN'],fontsize=15
                       ,rotation=45)
#axis[1].set_ylim([0,1.2])
axis[3].yaxis.set_tick_params(labelsize=15)
axis[3].legend(loc='best',prop=legend_properties)
axis[3].set_title('(d)',fontsize=15,fontweight = 'bold')
axis[3].set_ylabel('MAPE',rotation=90,fontsize=15)