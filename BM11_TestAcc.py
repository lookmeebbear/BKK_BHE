import math
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pathlib import Path
import subprocess as sp
from io import StringIO
import sklearn
from sklearn import linear_model, datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
from joblib import Parallel, delayed
import joblib

FILE = 'test_nDSM_100.csv' 

df = pd.read_csv(FILE)

lower_part = 0
upper_part = 50
print('=== cut 50 ===')

df = df.drop(columns=[ 'Unnamed: 0', 'inputmedian', 'geometry'])
df = df.dropna()
df = df.loc[df.height <= upper_part]

raster_band = ['nDSM_GTB_All_100.tif', 'nDSM_GTB_Op_100.tif', 'nDSM_GTB_SAR_100.tif',
                'nDSM_RF_All_100.tif', 'nDSM_RF_Op_100.tif', 'nDSM_RF_SAR_100.tif',
                'nDSM_SVR_All_100.tif','nDSM_SVR_OP_100.tif','nDSM_SVR_SAR_100.tif']


import pdb; pdb.set_trace()

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,mean_squared_error




def RANSAC(valdata) :
  X = valdata['Act'].to_numpy().reshape(-1,1)
  y = valdata['Pre'].to_numpy().reshape(-1,1)
  lr = linear_model.LinearRegression()
  lr.fit(X, y)

  # Robustly fit linear model with RANSAC algorithm
  ransac = linear_model.RANSACRegressor()
  ransac.fit(X, y)
  inlier_mask = ransac.inlier_mask_
  outlier_mask = np.logical_not(inlier_mask)

  # Predict data of estimated models
  line_X = np.arange(X.min(), X.max())[:, np.newaxis]
  line_y = lr.predict(line_X)
  line_y_ransac = ransac.predict(line_X)

  # Compare estimated coefficients
  print("Estimated coefficients (linear regression, RANSAC):")
  print(lr.coef_, ransac.estimator_.coef_)

  lw = 2
  x_45 = np.linspace(lower_part,upper_part,270)
  y_45 = x_45
  plt.plot(x_45, y_45, '-r', label = 'unit slope graph')
  plt.scatter(
      X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
  )
  plt.scatter(
      X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
  )
  plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")
  plt.plot(
      line_X,
      line_y_ransac,
      color="cornflowerblue",
      linewidth=lw,
      label="RANSAC regressor",
  )
 
  
  plt.legend(loc="upper left")
  plt.xlabel("Field Building Height (m)")
  plt.ylabel("Predicted Height (m)")
  plt.show()
  
  
def plot_graph(x, y, n):
  plt.scatter(x, y)
  plt.ylim([lower_part,n])
  plt.xlim([lower_part,n])
  x_45 = np.linspace(lower_part,n,upper_part)
  y_45 = x_45
  plt.plot(x_45, y_45, '-r', label = 'unit slope graph')
  #plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Field Building Height (m)')
  plt.ylabel('Predicted Height (m)')
  plt.legend()
  plt.show()
print('=========================')

print('Try with train data first')
print(" plot_graph(y_train,y_pred_rf_train,30) " )

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def plotden(x,y,lower_part,upper_part,mytitle):
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins=300
    k = gaussian_kde([x,y])
    xi, yi = np.mgrid[lower_part : upper_part : nbins*1j, lower_part :upper_part : nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
     
    # Make the plot
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
    
    
    x_45 = np.linspace(lower_part,upper_part,270)
    y_45 = x_45
    plt.plot(x_45, y_45, '-w', label = 'unit slope graph')
    
    X = x.to_numpy().reshape(-1,1)
    y = y.to_numpy().reshape(-1,1)
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(line_X)
    plt.plot(line_X, line_y, color="red", label="Linear regressor")
    plt.title(mytitle)
    plt.legend(loc="upper left")
    plt.xlabel('Field Building Height (m)')
    plt.ylabel('Predicted Height (m)')
    plt.show()
     
for k in range(1,10):
    print( raster_band[k-1] )
    myband = 'nDSM_'+str(k)
    y_train = df.height
    y_pred_rf_train = df[myband]
    plotden(y_train,y_pred_rf_train,lower_part,upper_part,raster_band[k-1])
    #import pdb; pdb.set_trace()
    
    
    r2_train = sklearn.metrics.r2_score(y_train, y_pred_rf_train)
    print('r2', r2_train)
    rmse_rf_train = math.sqrt( sklearn.metrics.mean_squared_error(y_train, y_pred_rf_train) )
    print('rmse', rmse_rf_train)
    valdata_df_train = pd.DataFrame( {'Act': y_train, 'Pre':y_pred_rf_train} )

    #RANSAC( valdata_df_train )
    #plot_graph(y_train,y_pred_rf_train,upper_part)
    print('============================================')


import pdb; pdb.set_trace()
