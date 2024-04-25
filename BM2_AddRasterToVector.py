# Building Height Estimation
# Prepare input data ... Add Satellite imagery data to building footprint
# Thepchai Srinoi
# Department of Survey Engineering

import geopandas as gpd
import rasterio as rio
from rasterstats import zonal_stats
import pandas as pd
import numpy as np
from tqdm import tqdm

# https://gis.stackexchange.com/questions/433246/zonal-statistics-spatial-join-raster-to-polygon-in-python

# Building Footprint 
zones = "building_inbound_DSM.gpkg"

# Satellite Image Raster data
values = "thesis_train_18082023.tif"

# I recommend .... DIVIDED AND CONQUER
# height consideration : more than this value
height_criteria = 20

#########################################################################
# Processing

# Import Geodataframe ... convert to UTM Zone 47
print('import geodataframe')
gdf = gpd.read_file(zones)
gdf = gdf.to_crs(32647)
#gdf = gdf.iloc[0:10]
print( gdf )

# Height Filtering
gdf = gdf.loc[ gdf.height > height_criteria]
gdf = gdf.reset_index(drop=True)
#print(gdf.head())

# Import DSM
print('import raster')
img = rio.open(values)
b_count = img.count
#import pdb; pdb.set_trace()

# Zonal Statistics ... Median Add Height to Vector Data
for myband in range(1,b_count+1) :
    print('band ...' , myband )
    print('rasterstat')
    stats = gpd.GeoDataFrame(zonal_stats(gdf, values, stats=["median"]
                                , band= myband))
    print('====================================')
    print( stats )
    
    print('add to dataframe')
    #gdf['median'] = stats['median']
    gdf[str(myband)] = stats
    print( gdf )
    #print( gdf.head() )

# Export Data
print('output')
gdf.to_file('traindata'+str(height_criteria) +'.gpkg', driver='GPKG')
gdf.to_csv('traindata'+str(height_criteria) +'.csv')
import pdb; pdb.set_trace()