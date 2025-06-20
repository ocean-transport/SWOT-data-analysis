import xarray as xr
import numpy as np
import glob
import zarr 

# Other Python libraries
import requests
import json

# Python standard library imports
from pprint import pprint

#turn off warnings
import warnings
warnings.filterwarnings("ignore")

# Import interpolation library
import sys
sys.path.append('../../SWOT-data-analysis/src')
import interp_utils

import sys
sys.path.append('/home/tm3076/projects/NYU_SWOT_project/SWOT-data-analysis/src')
import earthaccess
import swot_utils
from importlib import reload
import os
import traceback

# For sbatch script
import argparse

import timeit


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Global variables
DATA_PROVIDER = 'POCLOUD'

# Function to download and subset VIIRS satellite SST data using NASA Earthdata access
def download_raw_SST_earthaccess(data_short_name, save_path, sw_lon, sw_lat, ne_lat, ne_lon,
                                  start_time='2023-04-01T21:00:00Z', end_time='2023-07-28T20:59:59Z',
                                  only_get_pixel_mask=False, quality_level=1, save=False):
    """
    Downloads and subsets Sea Surface Temperature (SST) satellite data from NASA's 
    Common Metadata Repository (CMR) using the Earthaccess client.

    This function searches for data granules (e.g., VIIRS Level-2 products) within a 
    geographic bounding box and a specified time range. It then filters and extracts
    SST-related variables, optionally filtering by quality level, and saves the data 
    in NetCDF format.

    Parameters
    ----------
    data_short_name : str
        The short name of the dataset (e.g., "VIIRS_NPP-OSPO-L2P-v2.41") to retrieve.
    save_path : str
        Path to the local directory where the output NetCDF files will be saved.
    sw_lon : float
        Longitude of the southwest corner of the bounding box.
    sw_lat : float
        Latitude of the southwest corner of the bounding box.
    ne_lat : float
        Latitude of the northeast corner of the bounding box.
    ne_lon : float
        Longitude of the northeast corner of the bounding box.
    start_time : str, optional
        Start time (UTC) for the search period in ISO format. Default is 2023-04-01.
    end_time : str, optional
        End time (UTC) for the search period in ISO format. Default is 2023-07-28.
    only_get_pixel_mask : bool, optional
        If True, only retrieve and save a pixel mask based on `quality_level`. 
        If False, download full SST and associated quality data. Default is False.
    quality_level : int, optional
        The minimum acceptable quality level for filtering SST pixels. Currently unused 
        in this function but intended for future mask filtering logic. Default is 1.

    Returns
    -------
    list
        List of search result objects (metadata for matched data granules).
    """

    results = earthaccess.search_data(
        short_name=data_short_name,
        bounding_box=(sw_lon,sw_lat,ne_lat,ne_lon),
        temporal=(start_time,end_time),
        count=-1)

    if save:
        for result in results:
            file = earthaccess.open([result])[0]
            print(f"Found {file}, attempting download..")
            ds = xr.open_dataset(file)[["sea_surface_temperature","quality_level","l2p_flags"]].isel(time=0)
            if not os.path.exists(f"{save_path}/"):
                os.makedirs(f"{save_path}/",exist_ok=True)
            if os.path.exists(f"{save_path}/{file.full_name.split("/")[-1]}"):
                print(f"Some form of {save_path}/{file.full_name.split("/")[-1]} already exists! Skipping for now...")
            else:
                ds.to_netcdf(f"{save_path}/{file.full_name.split("/")[-1]}")
                
    return results



def subset_raw_SST_earthaccess(file_path, lat, lon, save_name="test", save_path="./tests/", n=128,
                                L_x=512e3, L_y=512e3, subset_deg=3,
                                timedelta_for_mean=np.timedelta64(1, 'D'),
                                only_get_pixel_mask=False, quality_level=2, skip_NaN=False, time_chunk=50):
    """
    Extract and process a spatial subset of SST (Sea Surface Temperature) satellite data
    from NetCDF files centered around a specified lat/lon coordinate. The data is interpolated
    to a regular ENU grid, optionally filtered for NaNs, and temporally averaged.

    Parameters
    ----------
    file_path : str
        Path to directory containing NetCDF files.
    lat : float
        Latitude of the region of interest.
    lon : float
        Longitude of the region of interest.
    save_name : str, optional
        Filename (without extension) for the output dataset. Default is "test".
    save_path : str, optional
        Directory where the processed dataset will be saved. Default is "./tests/".
    n : int, optional
        Resolution of the interpolated grid (number of grid points per axis). Default is 128.
    L_x : float, optional
        Physical width (in meters) of the interpolation grid in the x-direction. Default is 512e3.
    L_y : float, optional
        Physical height (in meters) of the interpolation grid in the y-direction. Default is 512e3.
    subset_deg : float, optional
        Spatial buffer in degrees around (lat, lon) for initial filtering. Default is 3.
    timedelta_for_mean : numpy.timedelta64 or None, optional
        Time interval for aggregating SST data. If None, no aggregation is performed.
        Default is 1 day (np.timedelta64(1, 'D')).
    only_get_pixel_mask : bool, optional
        If True, computes a maximum-based pixel mask instead of averaging SST values. Default is False.
    quality_level : int, optional
        Threshold for minimum acceptable quality level for SST pixels. Currently unused. Default is 1.
    skip_NaN : bool, optional
        If True, skips any NetCDF file containing more than 10 NaN quality_level values. Default is False.

    Returns
    -------
    ds_out : xarray.Dataset or None
        Merged and interpolated SST dataset, optionally temporally averaged.
        Returns None if no valid data was found or processed.
    """
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Check if the processed file already exists to avoid redundant computation
    if os.path.isfile(f"{save_path}{save_name}"):
        print(f"Some form of {save_path}{save_name}.zarr already exists! Skipping for now...")
        return
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Get all NetCDF files in the specified directory
    files = sorted(glob.glob(f"{file_path}/*.nc"))

    print("files[0]", files[0])
    print("files[1]", files[1])

    # Parse start and end dates from filenames
    start_date = files[0].split("/")[-1].split("-")[0]
    end_date = files[-1].split("/")[-1].split("-")[0]
    start_datetime = np.datetime64(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}")
    end_datetime = np.datetime64(f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}") + np.timedelta64(1, "D")

    print("start_date:", start_datetime)
    print("end_date:", end_datetime)

    # Parse data collection short name (assumed PO.DAAC-style directory)
    data_short_name = file_path.split("/")[-1].split("_pacific")[0]

    # Bound longitude within [-180, 180] range
    lon_min = max(lon - subset_deg, -180)
    lon_max = min(lon + subset_deg, 180)

    # Search for matching datasets from NASA Earthdata using earthaccess
    results = earthaccess.search_data(
        short_name=data_short_name,
        bounding_box=(lon_min, lat - subset_deg, lon_max, lat + subset_deg),
        temporal=(str(start_datetime), str(end_datetime)),
        count=-1)

    if len(results) < 1:
        print("No results found, aborting...")
    else:
        pprint(f"Found {len(results)} results!")

    # Extract matching filenames from the search results
    prefixes = [link.split("/")[-1] for result in results for link in result.data_links()]
    file_matches = [file for prefix in prefixes for file in files if prefix in file]

    arrs_out = []  # Store processed/interpolated datasets

    # Log file for skipped files
    with open(f"sbatch_logs/skipped_sst_files_for_{save_name}", "w") as f:
        f.write("skipped files \n------------- \n")

    # Process each matched file
    for file in file_matches:
        try:
            ds = xr.open_dataset(file)

            # Standardize lat/lon dimension names if necessary
            if "lat" in ds:
                ds = ds.rename({"lat": "latitude", "lon": "longitude"})

            # Skip file if the target region is clearly out of bounds
            if (lat < ds.latitude.min() - 3) or (lat > ds.latitude.max() + 3) or \
               (lon < ds.longitude.min() - 3) or (lon > ds.longitude.max() + 3):
                print("Likely no valid data in bounds, skipping file.")
                continue

            # Extract data subset around target lat-lon
            ds_subset = swot_utils.xr_subset(ds, [lat - subset_deg, lat + subset_deg],
                                                [lon - subset_deg, lon + subset_deg])

            if not isinstance(ds_subset, xr.Dataset):
                print("No valid data in bounds, skipping file.")
                continue
            
            # Do expensive filtering step here...
            #ds_out_sst_filtered_q2 = ds_subset.sea_surface_temperature.where(ds_subset.quality_level>=2,other=np.nan).rename("sst_filtered_q2")
            #ds_out_sst_filtered_q3 = ds_subset.sea_surface_temperature.where(ds_subset.quality_level>=3,other=np.nan).rename("sst_filtered_q3")
            #ds_out_sst_filtered_q4 = ds_subset.sea_surface_temperature.where(ds_subset.quality_level>=4,other=np.nan).rename("sst_filtered_q4")
            ds_out_sst_filtered_q5 = ds_subset.sea_surface_temperature.where(ds_subset.quality_level>=5,other=np.nan).rename("sst_filtered_q5")
            
            ds_subset = xr.merge([ds_subset,ds_out_sst_filtered_q5])
            # Interpolate to uniform ENU grid
            interp_ds = interp_utils.grid_everything(ds_subset, lat, lon, n=n, L_x=L_x, L_y=L_y)

            # Add time coordinate from the original subset
            interp_ds = interp_ds.expand_dims(dim={"time": 1}, axis=0).assign_coords(
                time=("time", [ds_subset.time.values]))

            # NaN filtering logic (optional)
            if skip_NaN:
                if interp_ds.quality_level.isnull().sum() > 10:
                    print(f"Significant NaN values found in {file}, skipping for now")
                else:
                    print(f"Successfully processed {file}, appending to dataset")
                    arrs_out.append(interp_ds)
            else:
                print(f"Successfully processed {file}, appending to dataset")
                arrs_out.append(interp_ds)

        except Exception as e:
            # Handle and log errors for debugging
            print(f"There was a problem handling {file}, skipping for now")
            with open(f"sbatch_logs/skipped_sst_files_for_{save_name}", "a") as f:
                f.write(f"{file}\n")
            traceback.print_exc()
            pass

    # Final dataset merge
    if len(arrs_out) < 1:
        ds_out = None
    else:
        ds_out = xr.concat(arrs_out, dim="time")

    # Optional daily aggregation (mean or max)
    if isinstance(timedelta_for_mean, type(None)):
        # Compute daily mean SST values
        pass
        
    elif isinstance(timedelta_for_mean, np.timedelta64):
        try:
            if only_get_pixel_mask:
                ds_out = ds_out.resample(time=timedelta_for_mean).max()
            else:
                # Compute daily mean SST values
                ds_out = ds_out.resample(time='1D').mean()
        except Exception as e:
            print(e)
            traceback.print_exc()
    else:
        print("timedelta_for_mean should be either None or np.timedelta64")

    # Notify user where the final dataset is stored
    if not isinstance(ds_out, type(None)):
        ds_out.chunk({'time': time_chunk}).to_zarr(f"{save_path}/{save_name}")
        print(f"Processed dataset saved at {save_path}/{save_name}")
                   
    return 

