import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from keras.models import load_model
from scipy.signal import savgol_filter as sg


model = load_model('D:\\Hassan\\Work\\milad\\perm prediction\\model_matrix\\fake models\\res2d.tf')
def model_predict(features_bw):
    return model.predict(features_bw)[:,0]

df_bw = pd.read_excel('D:\\Hassan\\Work\\milad\\perm prediction\\img_b\\data_bw.xlsx')
orig_df_bw = df_bw.copy()

target_bw = df_bw.pop('log(NMR_PERM)').values
target_bw_s = sg(target_bw,51,3)
target_bw = target_bw_s
index_bw = df_bw.pop('index')
features_bw = df_bw
features_bw['CGR'] = sg(features_bw['CGR'].values,31,3)
features_bw['NPHI'] = sg(features_bw['NPHI'].values,31,3)
features_bw['PEF'] = sg(features_bw['PEF'].values,31,3)
features_bw['RESD'] = sg(features_bw['RESD'].values,31,3)
features_bw['RESX'] = sg(features_bw['RESX'].values,31,3)
features_bw['RHOB'] = sg(features_bw['RHOB'].values,31,3)
features_bw['SGR'] = sg(features_bw['SGR'].values,31,3)

'''
# make a standard partial dependence plot
shap.partial_dependence_plot(
    "CGR", model.predict, features_bw, ice=False,
    model_expected_value=True, feature_expected_value=True
)
'''

background = shap.maskers.Independent(features_bw, max_samples=100)
explainer = shap.Explainer(model_predict, background)
shap_values = explainer(features_bw)

shap.plots.scatter(shap_values[:,'CGR'],
                   #color=shap_values
                   )

plt.figure()
shap.plots.beeswarm(shap_values)

clustering = shap.utils.hclust(features_bw, target_bw)
shap.plots.bar(shap_values)