# Building Height Estimation
# Prepare input data ... Add height to building footprint
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
zones = "building_inbound.gpkg"

# Raster data
values = "DSM_MEA.tif"


#########################################################################
# Processing

# Import Geodataframe ... convert to UTM Zone 47
print('import geodataframe')
gdf = gpd.read_file(zones)
gdf = gdf.to_crs(32647)
#gdf = gdf.iloc[0:10]
print( gdf )


# Import DSM
print('import raster')
img = rio.open(values)
b_count = img.count
#import pdb; pdb.set_trace()

# Zonal Statistics ... Median Add Height to Vector Data
stats = gpd.GeoDataFrame(zonal_stats(gdf, values, stats=["median"]
                                , band= 1))
print('add to dataframe')
gdf['_median'] = stats

# Calculate nDSM
print('Calculate Height above assumed ground')
gdf['height'] = gdf['_median'] - 30
print( gdf )
                                

# Export Data
print('output')
gdf.to_file("building_inbound_DSM.gpkg", driver='GPKG')

import pdb; pdb.set_trace()