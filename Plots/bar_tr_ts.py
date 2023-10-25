import matplotlib.pyplot as plt
import numpy as np
#%% R2
plt.subplot(131)
# Train
labels = [1,4,7]
heights = [0.9174,0.9978,0.9986]
plt.bar(labels,heights,label='Train Set')

# Test
labels = [2,5,8]
heights = [0.89283,0.997825,0.9985]
plt.bar(labels,heights,label='Test Set')

plt.bar([1.5,4.5,7.5],[0,0,0])
plt.xticks([1.5,4.5,7.5],
           [
            '1D-CNN',
            'Res 2D-CNN',
            'Multi-Input CNN'],rotation=40)
plt.legend()
plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel('R',rotation=90)
plt.title('(a) R')
#%% RMSE
plt.subplot(132)
# Train
labels = [1,4,7]
heights = [0.07984,0.0132292,0.01396]
plt.bar(labels,heights,label='Train Set')

# Test
labels = [2,5,8]
heights = [0.089192,0.0130759,0.014362]
plt.bar(labels,heights,label='Test Set')

plt.bar([1.5,4.5,7.5],[0,0,0])
plt.xticks([1.5,4.5,7.5],
           [
            '1D-CNN',
            'Res 2D-CNN',
            'Multi-Input CNN'],rotation=40)
plt.legend()
plt.yticks(np.arange(0,0.12,0.01))
plt.ylabel('RMSE',fontsize=10,rotation=90)
plt.title('(b) RMSE')
#%% MSE
plt.subplot(133)
# Train
labels = [1,4,7]
heights = [0.0063744,0.000175,0.000195]
plt.bar(labels,heights,label='Train Set')

# Test
labels = [2,5,8]
heights = [0.0079552,0.00017097,0.000206]
plt.bar(labels,heights,label='Test Set')

plt.bar([1.5,4.5,7.5],[0,0,0])
plt.xticks([1.5,4.5,7.5],
           [
            '1D-CNN',
            'Res 2D-CNN',
            'Multi-Input CNN'],rotation=40)
plt.legend()
plt.yticks(np.arange(0,0.012,0.001))
plt.ylabel('MSE',rotation=90)
plt.title('(c) MSE')