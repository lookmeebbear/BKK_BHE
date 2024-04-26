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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

import pickle
from joblib import Parallel, delayed
import joblib

scale = StandardScaler()

FILE = 'train.csv' 
OUTPUT_PIK = 'SVR_ALL_100.pkl'

df = pd.read_csv(FILE)
df = df.drop(columns='Unnamed: 0')
lower_part = 0
upper_part = 100
df = df.loc[df.height <= upper_part]

if 1 :
    print('SAR')
    df.drop(columns=[str(i) for i in range(13,91)], axis=1, inplace=True)

if 0:
    print('Optical')
    df.drop(columns=[str(i) for i in range(1,13)], axis=1, inplace=True)
    
df = df.dropna()
dfx = df.copy()
y = dfx.pop('height')
dfx.drop(columns=['_median','geometry'], axis=1, inplace=True)
x = dfx
import pdb; pdb.set_trace()

sc_X = StandardScaler()
sc_y = StandardScaler()
x = sc_X.fit_transform(x)
y = sc_y.fit_transform(y.to_numpy().reshape(-1,1))

#testing data size is of 20% of entire data
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.1, random_state=42)

from sklearn.metrics import make_scorer,mean_squared_error
mse = make_scorer(mean_squared_error,greater_is_better=False)

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [10,1, 0.1,0.01,0.001,0.0001],
              'kernel': ['linear', 'rbf']} 
  
grid = GridSearchCV(SVR(), param_grid,scoring=mse, cv=10, refit=True, verbose = 3)
  
# fitting the model for grid search
model_SVR = grid.fit(x_train, y_train)

# print best parameter after tuning
print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

#joblib.dump(grid, OUTPUT_PIK)

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

y_pred_svr_train = grid.predict(x_train)
y_pred_svr_train =  sc_y.inverse_transform( y_pred_svr_train.reshape(-1,1) )
y_train = sc_y.inverse_transform( y_train )

valdata_df_train = pd.DataFrame({"Act": (y_train).transpose()[0], "Pre": (y_pred_svr_train).transpose()[0]})
rmse_svr_train = math.sqrt( sklearn.metrics.mean_squared_error(y_train, y_pred_svr_train) )
print(rmse_svr_train)

#valdata_df_train.to_csv('training_data_svm.csv')
r2_train = sklearn.metrics.r2_score(y_train, y_pred_svr_train)
print(r2_train)

RANSAC( valdata_df_train )
print( '#plot_graph(y_train,y_pred_svr_train,30)' )

import pdb; pdb.set_trace()

#upper_part = 180
y_pred_svr = grid.predict(x_test)
y_pred_svr =  sc_y.inverse_transform( y_pred_svr.reshape(-1,1) )
y_test = sc_y.inverse_transform( y_test )

valdata_df = pd.DataFrame({"Act": (y_test).transpose()[0], "Pre": (y_pred_svr).transpose()[0]})
rmse_svr_test = math.sqrt( sklearn.metrics.mean_squared_error(y_test, y_pred_svr) )
print(rmse_svr_test)

#valdata_df_train.to_csv('training_data_svm.csv')
r2_train_t = sklearn.metrics.r2_score(y_test, y_pred_svr)
print(r2_train_t)

RANSAC( valdata_df )
print(' #plot_graph(y_test,y_pred_svr,30) ' )

print('export model ok')
#joblib.dump(grid, OUTPUT_PIK)
import pdb; pdb.set_trace()

