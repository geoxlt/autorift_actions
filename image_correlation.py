#! /usr/bin/env python

import xarray as xr
import rasterio as rio
import rioxarray
import numpy as np
import os
from autoRIFT import autoRIFT
from scipy.interpolate import interpn
import pystac_client
import odc.stac
import planetary_computer
import geopandas as gpd
from shapely.geometry import shape
import warnings
import argparse
import numpy as np

# silence some warnings from stackstac and autoRIFT
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def download_s2(img1_date, img2_date, aoi):
    '''
    Download a pair of Sentinel-2 images acquired on given dates over a given area of interest
    '''
    aoi_gpd = gpd.GeoDataFrame({'geometry':[shape(aoi)]}).set_crs(crs="EPSG:4326")
    crs = aoi_gpd.estimate_utm_crs()
    
    stac = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace)

    search = stac.search(intersects=aoi,
                         datetime=img1_date,
                         collections=["sentinel-2-l2a"])
    
    img1_items = search.item_collection()

    img1_ds = odc.stac.load(img1_items,
                            chunks={"x": 2048, "y": 2048},
                            bbox=aoi_gpd.total_bounds,
                            groupby='solar_day').where(lambda x: x > 0, other=np.nan)
    
    search = stac.search(intersects=aoi,
                         datetime=img2_date,
                         collections=["sentinel-2-l2a"])
    
    img2_items = search.item_collection()

    img2_ds = odc.stac.load(img2_items,
                            chunks={"x": 2048, "y": 2048},
                            bbox=aoi_gpd.total_bounds,
                            groupby='solar_day').where(lambda x: x > 0, other=np.nan)

    return img1_ds, img2_ds 

def run_autoRIFT(img1, img2, skip_x=2, skip_y=2, min_x_chip=16, max_x_chip=64,
                 preproc_filter_width=3, mpflag=4, search_limit_x=30, search_limit_y=30):
    '''
    Configure and run autoRIFT feature tracking with Sentinel-2 data for large mountain glaciers
    ''' 
    obj = autoRIFT()
    obj.MultiThread = mpflag

    obj.I1 = img1
    obj.I2 = img2

    obj.SkipSampleX = skip_x
    obj.SkipSampleY = skip_y

    # Kernel sizes to use for correlation
    obj.ChipSizeMinX = min_x_chip
    obj.ChipSizeMaxX = max_x_chip
    obj.ChipSize0X = min_x_chip
    # oversample ratio, balancing precision and performance for different chip sizes
    obj.OverSampleRatio = {obj.ChipSize0X:16, obj.ChipSize0X*2:32, obj.ChipSize0X*4:64}

    # generate grid
    m,n = obj.I1.shape
    xGrid = np.arange(obj.SkipSampleX+10,n-obj.SkipSampleX,obj.SkipSampleX)
    yGrid = np.arange(obj.SkipSampleY+10,m-obj.SkipSampleY,obj.SkipSampleY)
    nd = xGrid.__len__()
    md = yGrid.__len__()
    obj.xGrid = np.int32(np.dot(np.ones((md,1)),np.reshape(xGrid,(1,xGrid.__len__()))))
    obj.yGrid = np.int32(np.dot(np.reshape(yGrid,(yGrid.__len__(),1)),np.ones((1,nd))))
    noDataMask = np.invert(np.logical_and(obj.I1[:, xGrid-1][yGrid-1, ] > 0, obj.I2[:, xGrid-1][yGrid-1, ] > 0))

    # set search limits
    obj.SearchLimitX = np.full_like(obj.xGrid, search_limit_x)
    obj.SearchLimitY = np.full_like(obj.xGrid, search_limit_y)

    # set search limit and offsets in nodata areas
    obj.SearchLimitX = obj.SearchLimitX * np.logical_not(noDataMask)
    obj.SearchLimitY = obj.SearchLimitY * np.logical_not(noDataMask)
    obj.Dx0 = obj.Dx0 * np.logical_not(noDataMask)
    obj.Dy0 = obj.Dy0 * np.logical_not(noDataMask)
    obj.Dx0[noDataMask] = 0
    obj.Dy0[noDataMask] = 0
    obj.NoDataMask = noDataMask

    print("preprocessing images")
    obj.WallisFilterWidth = preproc_filter_width
    obj.preprocess_filt_lap() # preprocessing with laplacian filter
    obj.uniform_data_type()

    print("starting autoRIFT")
    obj.runAutorift()
    print("autoRIFT complete")

    # convert displacement to m
    obj.Dx_m = obj.Dx * 10
    obj.Dy_m = obj.Dy * 10
        
    return obj

def prep_outputs(obj, img1_ds, img2_ds):
    '''
    Interpolate pixel offsets to original dimensions, calculate total horizontal velocity
    '''

    # interpolate to original dimensions 
    x_coords = obj.xGrid[0, :]
    y_coords = obj.yGrid[:, 0]
    
    # Create a mesh grid for the img dimensions
    x_coords_new, y_coords_new = np.meshgrid(
        np.arange(obj.I2.shape[1]),
        np.arange(obj.I2.shape[0])
    )
    
    # Perform bilinear interpolation using scipy.interpolate.interpn
    Dx_full = interpn((y_coords, x_coords), obj.Dx, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dy_full = interpn((y_coords, x_coords), obj.Dy, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dx_m_full = interpn((y_coords, x_coords), obj.Dx_m, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dy_m_full = interpn((y_coords, x_coords), obj.Dy_m, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    
    # add variables to img1 dataset
    img1_ds = img1_ds.assign({'Dx':(['y', 'x'], Dx_full),
                              'Dy':(['y', 'x'], Dy_full),
                              'Dx_m':(['y', 'x'], Dx_m_full),
                              'Dy_m':(['y', 'x'], Dy_m_full)})
    # calculate x and y velocity
    img1_ds['veloc_x'] = (img1_ds.Dx_m/(img2_ds.time.isel(time=0) - img1_ds.time.isel(time=0)).dt.days)*365.25
    img1_ds['veloc_y'] = (img1_ds.Dy_m/(img2_ds.time.isel(time=0) - img1_ds.time.isel(time=0)).dt.days)*365.25
    
    # calculate total horizontal velocity
    img1_ds['veloc_horizontal'] = np.sqrt(img1_ds['veloc_x']**2 + img1_ds['veloc_y']**2)

    return img1_ds

def get_parser():
    parser = argparse.ArgumentParser(description="Run autoRIFT to find pixel offsets for two Sentinel-2 images")
    parser.add_argument("img1_date", type=str, help="date of first Sentinel-2 image ('YYYY-mm-dd')")
    parser.add_argument("img2_date", type=str, help="date of second Sentinel-2 image ('YYYY-mm-dd')")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # hardcoding a bbox for now
    aoi = {
          "type": "Polygon",
          "coordinates": [
              [
                  [79.2276515431187, 41.60190859421149],
                  [80.37541069846424, 41.60190859421149],
                  [80.37541069846424, 42.30137883933848],
                  [79.2276515431187, 42.30137883933848],
                  [79.2276515431187, 41.60190859421149]
              ]
          ]
      }

    # download Sentinel-2 images
    img1_ds, img2_ds = download_s2(args.img1_date, args.img2_date, aoi)
    # grab near infrared band only
    img1 = img1_ds.B08.squeeze().values
    img2 = img2_ds.B08.squeeze().values
    
    # scale search limit with temporal baseline assuming max velocity 1000 m/yr (100 px/yr)
    search_limit_x = search_limit_y = round(((((img2_ds.time.isel(time=0) - img1_ds.time.isel(time=0)).dt.days)*60)/365.25).item())
    
    # run autoRIFT feature tracking
    obj = run_autoRIFT(img1, img2, search_limit_x=search_limit_x, search_limit_y=search_limit_y)
    # postprocess offsets
    ds = prep_outputs(obj, img1_ds, img2_ds)

    # write out velocity to tif
    ds.veloc_horizontal.rio.to_raster(f'S2_{args.img1_date}_{args.img2_date}_horizontal_velocity.tif')

if __name__ == "__main__":
   main()
