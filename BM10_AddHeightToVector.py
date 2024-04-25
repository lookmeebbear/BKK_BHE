import geopandas as gpd
import rasterio as rio
from rasterstats import zonal_stats
import pandas as pd
import numpy as np
#from tqdm import tqdm
from shapely import wkt

# https://gis.stackexchange.com/questions/433246/zonal-statistics-spatial-join-raster-to-polygon-in-python

zones = "test.csv"

raster_band = ['nDSM_GTB_All_100.tif', 'nDSM_GTB_Op_100.tif', 'nDSM_GTB_SAR_100.tif',
                'nDSM_RF_All_100.tif', 'nDSM_RF_Op_100.tif', 'nDSM_RF_SAR_100.tif',
                'nDSM_SVR_All_100.tif','nDSM_SVR_OP_100.tif','nDSM_SVR_SAR_100.tif']

print('import geodataframe')
df = pd.read_csv('test.csv')
df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])

gdf = gpd.GeoDataFrame(df, geometry="geometry", crs='epsg:32647')
gdf = gdf.loc[ gdf.height >= 3]
gdf = gdf.loc[ gdf.height <= 100]
gdf = gdf[['inputmedian', 'height', 'geometry']]
gdf = gdf.reset_index(drop=True)
print( gdf )

import pdb; pdb.set_trace()

#b_count = img.count
k = 0
for values in raster_band :
    k += 1
    print('band ...' , values )
    
    print('rasterstat')
    stats = gpd.GeoDataFrame(zonal_stats(gdf, values, stats=["median"]
                                , band= 1))
    print('====================================')
    print( stats )
    
    print('add to dataframe')
    #gdf['output'] = stats['median']
    gdf['nDSM_'+ str(k) ] = stats
    #print( gdf )
    #print( gdf.head() )

print('output')
#gdf.to_file('traindata20.gpkg', driver='GPKG')
gdf.to_csv('test_nDSM_100.csv')
import pdb; pdb.set_trace()
