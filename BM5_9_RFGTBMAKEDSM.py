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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
from joblib import Parallel, delayed
import joblib
#650123.253,1510167.028, 676153.252,1543247.027
raster_file_all = 'thesis_big_east.tif'
OUTPUT_PIK = 'GTB_ALL_100.pkl'
OUTPUT = 'nDSM_GTB_All_east.tif'

model_ALL = joblib.load(OUTPUT_PIK)

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
                    img_preds = model.predict(to_predict)

                    # add the prediction back to the valid pixels (using only the first band of the mask to decide on validity)
                    # makes the assumption that all bands have identical no-data value arrangements
                    output = np.zeros(img_flat.shape[0])
                    #import pdb; pdb.set_trace()
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
