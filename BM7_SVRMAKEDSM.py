import math
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterstats.io import bounds_window
import rasterstats
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

from sklearn.metrics import r2_score
import pickle
from joblib import Parallel, delayed
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
scale = StandardScaler()


raster_file_all = 'thesis_train_18082023.tif'
OUTPUT_PIK = 'SVR_ALL_100.pkl'
OUTPUT = 'nDSM_SVR_ALL_100.tif'

sc_X = StandardScaler()
sc_y = StandardScaler()


##########################################################

FILE = 'train.csv' 

df = pd.read_csv(FILE)
df = df.drop(columns='Unnamed: 0')
lower_part = 0
upper_part = 100
df = df.loc[df.height <= upper_part]

if 0 :
    print('SAR')
    df.drop(columns=[str(i) for i in range(13,91)], axis=1, inplace=True)

if 0 :
    print('Optical')
    df.drop(columns=[str(i) for i in range(1,13)], axis=1, inplace=True)
    
df = df.dropna()
dfx = df.copy()
y = dfx.pop('height')
dfx.drop(columns=['_median','geometry'], axis=1, inplace=True)
x = dfx
#import pdb; pdb.set_trace()

sc_X = StandardScaler()
sc_y = StandardScaler()
sc_X.fit_transform(x)
sc_y.fit_transform(y.to_numpy().reshape(-1,1))

##########################################################

model_ALL = joblib.load(OUTPUT_PIK)
print("Let's go ...")


def makeDSM(model, new_image ,output_image):
    
    with rasterio.open(new_image, 'r') as src:
        profile = src.profile
        profile.update(
            dtype=rasterio.float64,
            count=1,
        )
        with rasterio.open(output_image, 'w', **profile) as dst:

            # perform prediction on each small image patch to minimize required memory
            patch_size = 500

            for i in range((src.shape[0] // patch_size) + 1):
                for j in range((src.shape[1] // patch_size) + 1):
                    # define the pixels to read (and write) with rasterio windows reading
                    window = rasterio.windows.Window(
                        j * patch_size,
                        i * patch_size,
                        # don't read past the image bounds
                        min(patch_size, src.shape[1] - j * patch_size),
                        min(patch_size, src.shape[0] - i * patch_size))
                    
                    # read the image into the proper format
                    data = src.read(window=window)

                    
                    # adding indices if necessary
                    img_swp = np.moveaxis(data, 0, 2)
                    img_flat = img_swp.reshape(-1, img_swp.shape[-1])

                    img_w_ind = np.concatenate([img_flat], axis=1)

                    # remove no data values, store the indices for later use
                    m = np.ma.masked_invalid(img_w_ind)
                    #to_predict = img_w_ind[~m.mask].reshape(-1, img_w_ind.shape[-1])
                    to_predict = np.nan_to_num(img_w_ind)
                    # to_predict = img_w_ind
                    # skip empty inputs
                    if not len(to_predict):
                        continue
                   
                    # predict
              
                    #import pdb; pdb.set_trace()
                    print(to_predict)
                    mp = model.predict( sc_X.fit_transform(to_predict) )
                    
                    print('Yes')
                    print( mp )
                    
                    img_preds = sc_y.inverse_transform( mp.reshape(-1,1) )

                    # add the prediction back to the valid pixels (using only the first band of the mask to decide on validity)
                    # makes the assumption that all bands have identical no-data value arrangements
                    output = np.zeros(img_flat.shape[0])
                    output = img_preds.flatten()
                    # resize to the original image dimensions
                    output = output.reshape(*img_swp.shape[:-1])

                    # create our final mask
                    #mask = (~m.mask[:, 0]).reshape(*img_swp.shape[:-1])

                    # write to the final files
                    dst.write(output.astype(rasterio.float64), 1, window=window)
                    #dst.write_mask(mask, window=window)
    print('Output finish ', output_image)
    print('=================================')


#raster_file_all = "stack_satelliteimage_opensky_kernel1.tif"
makeDSM(model_ALL,raster_file_all, OUTPUT)


import pdb; pdb.set_trace()

