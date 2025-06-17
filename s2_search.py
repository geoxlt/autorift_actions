import xarray as xr
import rasterio
import rioxarray
import os
import pystac_client
import json
import pandas as pd
import argparse
import odc.stac
import planetary_computer
import geopandas as gpd
from shapely.geometry import shape
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description="Search for Sentinel-2 images")
    parser.add_argument("cloud_cover", type=str, help="percent cloud cover allowed in images (0-100)")
    parser.add_argument("start_month", type=str, help="first month of year to search for images")
    parser.add_argument("stop_month", type=str, help="last month of year to search for images")
    parser.add_argument("npairs", type=str, help="number of pairs per image")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # hardcode bbox for now
    aoi = {
        "type": "Polygon",
        "coordinates": [
            [[-121.85308124466923,46.94373134954458],
            [-121.85308124466923,46.785391494446145],
            [-121.63845508457872,46.785391494446145],
            [-121.63845508457872,46.94373134954458],
            [-121.85308124466923,46.94373134954458]]]
    }

    aoi_gpd = gpd.GeoDataFrame({'geometry':[shape(aoi)]}).set_crs(crs="EPSG:4326")
    crs = aoi_gpd.estimate_utm_crs()
    
    stac = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace)

    # search planetary computer
    search = stac.search(
        intersects=aoi,
        datetime="2021-01-01/2025-06-17",
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": float(args.cloud_cover)}})

    items = search.item_collection()
    
    s2_ds = odc.stac.load(items,chunks={"x": 2048, "y": 2048},
                          bbox=aoi_gpd.total_bounds,
                          groupby='solar_day').where(lambda x: x > 0, other=np.nan)
    print(f"Returned {len(s2_ds.time)} acquisitions")
    
    # calculate number of valid pixels in each image
    total_pixels = len(s2_ds.y)*len(s2_ds.x)
    nan_count = (~np.isnan(s2_ds.B08)).sum(dim=['x', 'y']).compute()
    # keep only pixels with 90% or more valid pixels
    s2_ds = s2_ds.where(nan_count >= total_pixels*0.9, drop=True)

    # filter to specified month range
    s2_ds_snowoff = s2_ds.where((s2_ds.time.dt.month >= int(args.start_month)) & (s2_ds.time.dt.month <= int(args.stop_month)), drop=True)

    # get dates of acceptable images
    image_dates = s2_ds_snowoff.time.dt.strftime('%Y-%m-%d').values.tolist()
    print('\n'.join(image_dates))
    
    # Create Matrix Job Mapping (JSON Array)
    pairs = []
    for r in range(len(s2_ds_snowoff.time) - int(args.npairs)):
        for s in range(1, int(args.npairs) + 1 ):
            t_baseline = s2_ds_snowoff.isel(time=r+s).time - s2_ds_snowoff.isel(time=r).time
            if t_baseline.dt.days <= 100: #t baseline threshold
                img1_date = image_dates[r]
                img2_date = image_dates[r+s]
                shortname = f'{img1_date}_{img2_date}'
                pairs.append({'img1_date': img1_date, 'img2_date': img2_date, 'name':shortname})
    matrixJSON = f'{{"include":{json.dumps(pairs)}}}'
    print(f'number of image pairs: {len(pairs)}')
    
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        print(f'IMAGE_DATES={image_dates}', file=f)
        print(f'MATRIX_PARAMS_COMBINATIONS={matrixJSON}', file=f)

if __name__ == "__main__":
   main()
