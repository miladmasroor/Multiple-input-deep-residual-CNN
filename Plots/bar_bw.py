import matplotlib.pyplot as plt
import numpy as np
#%% R2
plt.subplot(131)
# Blind
labels = [1,2,3]
heights = [0.91794,0.97412,0.998852]
plt.bar(labels,heights,label='Blind Well')

plt.xticks(labels,
           ['1D-CNN',
            'Res 2D-CNN',
            'Multi-Input CNN'],rotation=40)
plt.legend()
plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel('R',rotation=90)
plt.title('(a) R')
#%% RMSE
plt.subplot(132)
# Blind
labels = [1,2,3]
heights = [0.090352,0.042558,0.01346]
plt.bar(labels,heights,label='Blind Well')

plt.xticks(labels,
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
# Blind
labels = [1,2,3]

heights = [0.008163,0.001811,0.000181]
plt.bar(labels,heights,label='Blind Well')

plt.xticks(labels,
           [
            '1D-CNN',
            'Res 2D-CNN',
            'Multi-Input CNN'],rotation=40)
plt.legend()
plt.yticks(np.arange(0,0.012,0.001))
plt.ylabel('MSE',rotation=90)
plt.title('(c) MSE')