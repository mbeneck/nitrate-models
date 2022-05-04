# -*- coding: utf-8 -*-
"""
Created on Mon May  2 17:25:35 2022

@author: eckmb
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def ninefold_linear_fit(X, y):
    x_surf = np.linspace(30, 35, num=50)
    y_surf = np.linspace(0, 25, num=50)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    surf_df = pd.DataFrame({'sss':x_surf.ravel(), 'sst':y_surf.ravel()})
    
    result_df = pd.DataFrame(columns=['SSS coef', 'SST coef', 'Intercept', 'R2'])
    fig, axs= plt.subplots(nrows= 3, ncols=3, subplot_kw={'projection':'3d'}, figsize=(8.5,  9.5))
    kf = KFold(n_splits=9)
    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        reg = LinearRegression().fit(X_train, y_train)
        plane = reg.predict(surf_df)
        
        score = reg.score(X_test, y_test)
        
        result_df.loc[idx] = [reg.coef_[0], reg.coef_[1], reg.intercept_, score]
        
        ax=axs.flatten()[idx]
        ax.scatter(X_test['sst'],X_test['sss'], y_test, alpha=1, c='blue', marker='o')
        ax.plot_wireframe(y_surf,x_surf, plane.reshape(x_surf.shape),  rstride= 5, cstride=5, alpha=.4)
        ax.view_init(elev=20,azim=30)
        ax.set_xlabel('SST')
        ax.set_ylabel('SSS')
        ax.set_zticks(np.arange(0, 15, 2.5))
        if(idx % 3 == 0):
            ax.set_zlabel('Nitrate (umol)')
        else:
            ax.set_zticklabels([])
            
        ax.set_title('Fold %i, R2 = %.2f' % (idx+1, score))
        
    return fig, axs, result_df

E1 = pd.read_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\Processed Data\\E1_combined.csv', index_col=0, parse_dates=True)
M1 = pd.read_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\Processed Data\\M1_combined.csv', index_col=0, parse_dates=True)
N1 = pd.read_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\Processed Data\\N1_combined.csv', index_col=0, parse_dates=True)

combined = pd.concat([E1[E1['depth'] == 50], M1[M1['depth'] == 50], N1[N1['depth'] == 50]])

# randomizes row order and removes time index
combined_rand = combined.sample(frac=1).reset_index(drop=True)

features = ['sss', 'sst']
target = 'Nitrate_umol'

X_alldata = combined_rand[features]
y_alldata = combined_rand[target]

fig_alldata, axs_alldata, result_df_alldata = ninefold_linear_fit(X_alldata,y_alldata)
fig_alldata.suptitle('Linear Model Fit Results for 50m Nitrate at Buoys E, M, N (NM1)', fontsize= 18)

M1_50_rand = M1.loc[M1['depth'] == 50].sample(frac=1).reset_index(drop=True)
M1_1_rand = M1.loc[M1['depth'] == 1].sample(frac=1).reset_index(drop=True)

fig_M1_50, axs_M1_50, result_df_M1_50 = ninefold_linear_fit(M1_50_rand[features], M1_50_rand[target])
fig_M1_1, axs_M1_1, result_df_M1_1 = ninefold_linear_fit(M1_1_rand[features], M1_1_rand[target])

fig_M1_50.suptitle('Linear Model Fit Results for 50m Nitrate at Buoy M (NM2)', fontsize= 18)
fig_M1_1.suptitle('Linear Model Fit Results for 1m Nitrate at Buoy M (NM3)', fontsize= 18)


for ax in axs_M1_1.flatten():
    ax.set_zticks(np.arange(-10, 25, 5))

M1_compare = pd.DataFrame()
M1_compare['50m Nitrate (umol)'] = M1.loc[M1['depth'] == 50]['Nitrate_umol']
M1_compare['1m Nitrate (umol)'] = M1.loc[M1['depth'] == 1]['Nitrate_umol']
M1_fig, M1_ax = plt.subplots()
M1_compare.plot(x='1m Nitrate (umol)', y='50m Nitrate (umol)', ax=M1_ax, kind='scatter', title= '50m Nitrate vs 1m Nitrate at Buoy M')

