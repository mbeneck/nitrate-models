# -*- coding: utf-8 -*-
"""
Created on Tue May  3 22:49:11 2022

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
    x_surf = np.linspace(30, 34, num=50)
    y_surf = np.linspace(-5, 25, num=50)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    surf_df = pd.DataFrame({'salinity':x_surf.ravel(), 'sst':y_surf.ravel()})
    
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
        ax.scatter(X_test['sst'],X_test['salinity'], y_test, alpha=1, c='blue', marker='o')
        ax.plot_wireframe(y_surf,x_surf, plane.reshape(x_surf.shape),  rstride= 5, cstride=5, alpha=.4)
        ax.view_init(elev=20,azim=70)
        ax.set_xlabel('SST')
        ax.set_ylabel('SSS')
        ax.set_ylim(bottom=30, top=33)
        ax.set_zticks(np.arange(0, 20, 4))
        ax.set_yticks(np.arange(30, 34, 1))
        if(idx % 3 == 0):
            ax.set_zlabel('Nitrate (umol)')
        else:
            ax.set_zticklabels([])
            
        ax.set_title('Fold %i, R2 = %.2f' % (idx+1, score))
        
    return fig, axs, result_df

GOM_modis = pd.read_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\Processed Data\\gom_shallow_ave_modis.csv', index_col=0, parse_dates=True)
GOM_modis.dropna(inplace=True, subset=['sst', 'temp'])

#randomly shuffle data
GOM_modis_rand = GOM_modis.sample(frac=1).reset_index(drop=True)

features = ['salinity', 'sst']
target = 'Nitrate'


# sanity check modis temp vs in-situ temp
vals = np.linspace(0, 25, 50)
reg = LinearRegression().fit(GOM_modis['sst'].values.reshape(-1, 1), GOM_modis['temp'].values.reshape(-1, 1))
line = reg.predict(vals.reshape(-1, 1))

score = reg.score(GOM_modis['sst'].values.reshape(-1, 1), GOM_modis['temp'].values.reshape(-1, 1))
fig1, ax1 = plt.subplots()
GOM_modis.plot(x='sst', y='temp', kind='scatter', title = 'MODIS SST vs In-Situ SST', ax=ax1)
ax1.set_ylabel('In-situ Temp (C)')
ax1.set_xlabel('MODIS SST (C)')
ax1.plot(vals,line, color = 'red')
ax1.text(0, 15, 'Slope: %.2f' % (reg.coef_[0]))
ax1.text(0, 12.5, 'Intercept: %.2f' % (reg.intercept_))
ax1.text(0, 10, 'R2: %.4f' % (score))

fig_GOM, axs_GOM, result_df_GOM = ninefold_linear_fit(GOM_modis_rand[features], GOM_modis_rand[target])
fig_GOM.suptitle('Linear Model Fit Results for Rebuck GOM Dataset (RM1)', fontsize= 18)

fig2, ax2= plt.subplots()
fig3, ax3= plt.subplots()

GOM_modis.plot(x='sst', y='Nitrate', kind='scatter', title='MODIS SST vs Nitrate')
GOM_modis.plot(x='salinity', y='Nitrate', kind='scatter', title='In-situ Salinity vs Nitrate', ax=ax2)
ax2.set_xlim(left=29, right=35)
GOM_modis.plot(x='sst', y='salinity', kind='scatter', title='MODIS SST vs In-situ Salinity', ax=ax3)
ax3.set_ylim(bottom=29, top=35)

GOM_modis_results= GOM_modis
allGOM_reg = LinearRegression().fit(GOM_modis[features], GOM_modis[target])
GOM_modis_results['prediction'] = allGOM_reg.predict(GOM_modis[features])
GOM_modis_results['pce'] = (GOM_modis_results['prediction'] - GOM_modis_results['Nitrate'])/GOM_modis_results['Nitrate']*100

GOM_modis_results.replace([np.inf, -np.inf], np.nan, inplace=True)
GOM_modis_results.dropna(subset=['pce'], how='any', inplace=True)

GOM_modis_results.to_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\Processed Data\\gom_modis_results.csv')