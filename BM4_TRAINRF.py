## LOOKMEE COMEBACK .... loss band 50

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

FILE = 'train.csv' 
OUTPUT_PIK = 'RF_ALL_100.pkl'

df = pd.read_csv(FILE)


lower_part = 0
upper_part = 100
df = df.loc[df.height <= upper_part]

if 0 :
    print('SAR')
    df.drop(columns=[str(i) for i in range(13,91)], axis=1, inplace=True)
if 0 :
    print('Optical')
    df.drop(columns=[str(i) for i in range(1,13)], axis=1, inplace=True)

#df.dropna(subset=['1'], inplace=True) 
dfx = df.copy()
#dfx.drop(columns=[['Unnamed: 0', '_median', 'geometry']], axis=1, inplace=True)
dfx = dfx.drop(columns=[ 'Unnamed: 0', '_median', 'geometry'])
dfx = dfx.dropna()
x = dfx
y = dfx.pop('height')

import pdb; pdb.set_trace()
print(len(x))
#import pdb; pdb.set_trace()

x_train, x_test, y_train, y_test =train_test_split(x,y, test_size= 0.1, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,mean_squared_error
mse = make_scorer(mean_squared_error,greater_is_better=False)

# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth' : [100,200,300,400,500,None],
    'max_features': [0.3,1.0,'sqrt','log2'],
    'n_estimators': [100,250,500,600,750,1000]
    }
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid = GridSearchCV(estimator = rf, param_grid = param_grid,scoring=mse,
                            cv = 10, n_jobs = -1, verbose = 3)
                      

#Train the model on training data
model_rf = grid.fit(x_train, y_train)
print(grid.best_params_)

print('export model ok')
joblib.dump(grid, OUTPUT_PIK)


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
# Use the forest's predict method on the test data
y_pred_rf_train = model_rf.predict(x_train)

#valdata_rf = pd.DataFrame({"Actual": ((y_test.to_numpy()).transpose())[0], "Predict": y_pred_rf})
valdata_df_train = pd.DataFrame( {'Act': y_train.to_numpy(), 'Pre':y_pred_rf_train} )
r2_train = sklearn.metrics.r2_score(y_train, y_pred_rf_train)
print('r2', r2_train)
rmse_rf_train = math.sqrt( sklearn.metrics.mean_squared_error(y_train, y_pred_rf_train) )
print('rmse', rmse_rf_train)

valdata_df_train.to_csv('training_data_val.csv')


RANSAC( valdata_df_train )
plot_graph(y_train,y_pred_rf_train,upper_part)
import pdb; pdb.set_trace()

print('=========================')
###############################################################################
print('Continue try with test data')
print(" plot_graph(y_test,y_pred_rf,60) " )
# Use the forest's predict method on the test data
y_pred_rf = model_rf.predict(x_test)
#valdata_rf = pd.DataFrame({"Actual": ((y_test.to_numpy()).transpose())[0], "Predict": y_pred_rf})
valdata_df = pd.DataFrame( {'Act': y_test.to_numpy(), 'Pre':y_pred_rf} )
mse_rf = sklearn.metrics.mean_squared_error(y_test, y_pred_rf)
r2 = sklearn.metrics.r2_score(y_test, y_pred_rf)
print('R2 score', r2)
#mae_rf = sklearn.metrics.mean_absolute_error(y_test, y_pred_rf)
print(math.sqrt( mse_rf) )
print('=========================')
#print( mae_rf )
RANSAC( valdata_df )

plt.scatter(y_test,y_pred_rf)
plt.show()
import pdb; pdb.set_trace()

n = 100
plot_graph(y_test,y_pred_rf,upper_part)
import pdb; pdb.set_trace()

imp = grid.best_estimator_.feature_importances_
col_list = x.columns.to_list()
dd = pd.DataFrame({'FD':imp , 'col': col_list[:-1]})
ddx = (dd.sort_values(by='FD',ascending=False))
ddx.to_csv('var_imp.csv')
print( ddx )
print( ddx['col'].to_list() )

print('=========================')
import pdb; pdb.set_trace()
