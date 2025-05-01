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
                                only_get_pixel_mask=False, quality_level=1, skip_NaN=False):
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
        pass
    elif isinstance(timedelta_for_mean, np.timedelta64):
        try:
            if only_get_pixel_mask:
                ds_out = ds_out.resample(time=timedelta_for_mean).max()
            else:
                # Compute daily mean SST values
                ds_out_quality_level = ds_out.quality_level
                ds_out_sst = ds_out.sea_surface_temperature
                ds_out_sst_filtered = ds_out_sst.where(ds_out_quality_level>=2,other=np.nan)
                ds_out_sst_filtered = ds_out_sst_filtered.resample(time='1D').mean().rename("sea_surface_temperature_filtered")
                ds_out = ds_out.resample(time='1D').mean()
                ds_out = xr.merge((ds_out,ds_out_sst_filtered))
        except Exception as e:
            print(e)
            traceback.print_exc()
    else:
        print("timedelta_for_mean should be either None or np.timedelta64")

    # Notify user where the final dataset is stored
    if not isinstance(ds_out, type(None)):
        ds_out.to_zarr(f"{save_path}/{save_name}")
        print(f"Processed dataset saved at {save_path}/{save_name}")
                   
    return 



'''
# DEPCREIATED

def subset_raw_SST(file_path, lat, lon, save_name="test", save_path="./tests/", n=256, L_x=512e3, L_y=512e3, subset_deg=3,
                     timedelta_for_mean=None, only_get_pixel_mask=False, quality_level=1, skip_NaN=False):
    """
    This function extracts a spatial subset from a set of satellite observations
    centered on a given latitude and longitude, interpolates them
    onto a standard ENU grid centered at the user-provided lat-lon coordinates, 
    and merges them into a single dataset. You can optionally subset by time.
    
    Parameters
    ----------
    file_path : str
        Path to the directory containing NetCDF files.
    lat : float
        Latitude of the target location.
    lon : float
        Longitude of the target location.
    save_name : str, optional
        Name of the output file (default is "test").
    save_path : str, optional
        Directory where the output file will be saved (default is "./tests/").
    n : int, optional
        Grid resolution for interpolation (default is 256).
    L_x : float, optional
        X-dimension length scale for interpolation (default is 512 km).
    L_y : float, optional
        Y-dimension length scale for interpolation (default is 512 km).
    subset_deg : float, optional
        Initial subset range in degrees (default is 3 degrees around target location).
    timedelta_for_mean : np.timedelta64 or None, optional
        Time interval for computing mean SST values (default is None, meaning no averaging).
    only_get_pixel_mask : bool, optional
        If True, computes a pixel mask instead of averaging (default is False).
    quality_level : int, optional
        Minimum quality level threshold for filtering data (default is 1).
    skip_NaN : bool, optional
        If True, skip any 
    
    Returns
    -------
    ds_out : xarray.Dataset
        Processed SST dataset with spatial subsetting, interpolation, and optional time averaging.
    """
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Check if the processed file already exists to avoid redundant computation
    if os.path.isfile(f"{save_path}{save_name}"):
        print(f"Some form of {save_path}{save_name}.nc already exists! Skipping for now...")
        return
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Get all NetCDF files in the specified directory
    files = sorted(glob.glob(f"{file_path}/*.nc"))

    # Find the start and datetime to search...
    start_date = files[0].split("/")[5].split("-")[0]
    end_date = files[-1].split("/")[5].split("-")[0]
    
    start_datetime = np.datetime64(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}")
    end_datetime = np.datetime64(f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}")+np.timedelta64(1,"D")

    
    # Use PO.DAAC to search for overlapping files
    cmr_url = 'https://cmr.earthdata.nasa.gov/search/granules.json'

    # Hacky fix here to make sure you're not outside of [-180,180] longitude
    lon_min = lon-subset_deg
    lon_max = lon+subset_deg
    if lon-subset_deg < -180:
        lon_min = -180
    if lon+subset_deg > 180:
        lon_max = 180
        
    # Request granule metadata from the CMR API based on dataset name and search bounds
    response = requests.get(cmr_url, 
                            params={
                                'provider': DATA_PROVIDER,
                                'short_name': "AVHRRF_MA-STAR-L2P-v2.80", 
                                'temporal': f'{start_datetime},{end_datetime}',
                                # If you try putting in a longitude outside of [-180,180] you get an error:
                                # {'errors': ['East must be within [-180.0] and [180.0] but was [181.0].']}
                                'bounding_box': f'{lon_min},{lat-subset_deg},{lon_max},{lat+subset_deg}',
                                'page_size': 2000,
                                }
                           )
    try:
        granules = response.json()['feed']['entry']
    except Exception as e:
        # Handle any errors during processing and log the traceback
        print("Lat lon error?")
        print(f'{max(lon-subset_deg,-180)},{lat-subset_deg},{min(lon-subset_deg,180)},{lat+subset_deg}')
        print(e)
        print(response.json())
        traceback.print_exc()

        
    pprint(f"Found {len(granules)} granules!")
    
    if len(granules) < 1:
        print("No granules found, aborting...")
       
    
    prefixes = []
    for granule in granules:
        for link in granule['links']:
            if link['rel'].endswith('/data#'):
                url = link['href']
                prefixes.append(url.split("/")[-1].split("-")[0])
                break

    
    file_matches = []
    for prefix in prefixes:
        for file in files:
            if prefix in file:
                file_matches.append(file)
            
    # Initialize a list to store processed datasets
    arrs_out = []
    
    # Loop through each NetCDF file
    for file in file_matches:
        try:
            # Open the NetCDF dataset
            ds = xr.open_dataset(file)
            
            # Ensure latitude and longitude dimensions are correctly named
            if "lat" in ds.dims:
                ds = ds.rename({"lat": "latitude", "lon": "longitude"})
                
            # Perform an initial lat-lon filter to avoid loading unnecessary data.
            if (lat < ds.latitude.min()-3) or (lat > ds.latitude.max()+3) \
               or (lon < ds.longitude.min()-3) or (lon > ds.longitude.max()+3):
                print("Likely no valid data in bounds, skipping file.")
                continue
                
            # Extract a spatial subset of the dataset centered on the target location
            ds_subset = swot_utils.xr_subset(ds, [lat-subset_deg, lat+subset_deg], [lon-subset_deg, lon+subset_deg])
            
            # If the subset is empty, skip further processing
            if not isinstance(ds_subset, xr.Dataset):
                print("No valid data in bounds, skipping file.")
                continue
       
            # Interpolate the subset onto a standardized grid
            interp_ds = interp_utils.grid_everything(ds_subset, lat, lon,  n=n, L_x=L_x, L_y=L_y)            
            # Add a time dimension to the dataset
            interp_ds = interp_ds.expand_dims(dim={"time": 1},axis=0).assign_coords(time = ("time", [ds_subset.time.values]))

            # Awkward block to handle NaN values, I added the nested "if" statements 
            # so I don't need perform the isnull() calc unless I want to look for NaNs.
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
            # Handle any errors during processing and log the traceback
            print(e)
            traceback.print_exc()
            return # REMOVE FOR FINAL VERSION

    # Skip empty datasets (for example if contaminated by NaN values)
    if len (arrs_out) < 1:
        ds_out = None
    else:
        # Concatenate all processed datasets along the time dimension
        ds_out = xr.concat(arrs_out,dim="time")
    
    # Handle time averaging if requested
    if isinstance(timedelta_for_mean, type(None)):
        pass  # No time averaging needed
    elif isinstance(timedelta_for_mean, np.timedelta64):
        try:
            if only_get_pixel_mask:
                # Compute maximum values for pixel mask (binary values at each time step)
                ds_out = ds_out.resample(time='1D').max()
            else:
                # Compute daily mean SST values
                ds_out = ds_out.resample(time='1D').mean()
        except Exception as e:
            # Handle any errors during processing and log the traceback
            print(e)
            traceback.print_exc()
            print(ds_out)
            return # REMOVE FOR FINAL VERSION
    else:
        print("timedelta_for_mean should be either None or np.timedelta64")

    if not isinstance(ds_out, type(None)):
        print(f"Processed dataset saved at {save_path}{save_name}")

    return ds_out
    
'''


"""
DEMO
############################################################
# Argument parser to do sepcific cycles
############################################################
# I really don't know what I'm doing here...
# See https://docs.python.org/dev/library/argparse.html
parser = argparse.ArgumentParser(prog="Program name",
                                 description="What do I do?",
                                 epilog="Like tears in the rain")

parser.add_argument("data_short_name")
parser.add_argument("save_path")
parser.add_argument("lat", type=float)
parser.add_argument("lon", type=float)
parser.add_argument("save_name")
parser.add_argument("n", type=int)
parser.add_argument("L_x", type=float)
parser.add_argument("L_y", type=float)
parser.add_argument("start_time")
parser.add_argument("end_time")

args = parser.parse_args()
print()

print(str(args.data_short_name))
print(str(args.save_path))
print(float(args.lat))
print(float(args.lon))
print(str(args.save_name))
print(int(args.n))
print(float(args.L_x))
print(float(args.L_y))
print(args.start_time)
print(args.end_time)

download_subset_VIIRS(str(args.data_short_name),
                      str(args.save_path),
                      args.lat,
                      args.lon,
                      str(args.save_name),
                      args.n,
                      args.L_x,
                      args.L_y,
                      start_time = args.start_time,
                      end_time = args.end_time,
                     )

"""


"""
OLD VERSION USING EOSDISTORE
# Function to download and subset VIIRS satellite data
def download_subset_grid_SST(data_short_name, save_path, lat, lon, name="", n=256, L_x=256e3, L_y=256e3,
                         start_time='2023-04-01T21:00:00Z', end_time='2023-07-28T20:59:59Z',
                         timedelta_for_mean=None, only_get_pixel_mask=False, quality_level=1, filter=False):
    '''
    Downloads and subsets SST satellite data from NASA's CMR (Common Metadata Repository).
    The function searches for granules (data files) within a specified latitude/longitude range
    and filters the data based on quality control measures. Finally, the processed data is
    saved as a NetCDF file for further analysis.

    Most of the highres SST is from two infrared instrument types, the Advanced Very High 
    Resolution Radiometer (AVHRRF) and the Visible Infrared Imaging Radiometer Suite (VIIRS), 
    which have been used on multiple NOAA and European MetOp satellites. The AVHRRF data we 
    have access to appears to be from the MetOp-A, -B, and -C satellites (Oct 2007 - Nov 2021, 
    Sep 2012 - present, Nov 2018 - present) and VIIRS data from the NOAA S-NPP, NOAA-20, and 
    NOAA-2021 satellites (2011 - present, 2017 - present, 2022 - present).

    

    Parameters
    ----------
    data_short_name : str
        Short name of the dataset to be retrieved from NASA's CMR.
    save_path : str
        Directory path where the subsetted data files will be stored.
    lat : float
        Central latitude for the data search.
    lon : float
        Central longitude for the data search.
    name : str, optional
        A custom name tag to append to the saved file names.
    n : int, optional
        Grid size for interpolating the final dataset (default is 256).
    L_x : float, optional
        Grid resolution in the x-direction (default is 256 km, 
        corresponding to 1km x-resolution for n=256).
    L_y : float, optional
        Grid resolution in the y-direction (default is 256 km, 
        corresponding to 1km y-resolution for n=256).
    start_time : str, optional
        The UTC start time to bounding the search. Default is
        a little before the first day of the 1-day SWOT repeat phase.
    end_time : str, optional
        The UTC end time to bounding the search. Default is
        the a little after the last day of the 1-day SWOT repeat phase.
    timedelta_for_mean : numpy timedelta64, optional
        The time delta over which to calculate mean field values. If "None", skip
        averaging and return individual observations. Example 
    only_get_pixel_mask : bool, optional
        Boolean flag to instruct the script to (False) download sst, quality_flags,
        and filtered sst or (True) download a boolean pixel mask based on the "quality_level"
        parameter below. The latter option is good for saving memory if you just want the 
        pixel mask.
    quality_level : float, optional
        
        
    Returns
    -------
    None
        Saves subsetted data as either NetCDF (only_get_pixel_mask = False) 
        or .npy (only_get_pixel_mask = True) files in the save directory.
    '''
    #start_timeit = timeit.default_timer()
    # code you want to evaluate
    #elapsed = timeit.default_timer() - start_timeit
    
    # Define the NASA CMR API URL for searching granules
    cmr_url = 'https://cmr.earthdata.nasa.gov/search/granules.json'
    
    # Request granule metadata from the CMR API based on dataset name and search bounds
    response = requests.get(cmr_url, 
                            params={
                                'provider': DATA_PROVIDER,
                                'short_name': data_short_name, 
                                'temporal': f'{start_time},{end_time}',
                                'bounding_box': f'{lon-2},{lat-2},{lon+2},{lat+2}',
                                'page_size': 2000,
                                }
                           )
    #print("line 119", timeit.default_timer() - start_timeit)

    # Extract the list of granules from the response JSON
    granules = response.json()['feed']['entry']
    pprint(f"Found {len(granules)} granules!")
    if len(granules) < 1:
        print("No granules found, aborting...")
        return    
        
    # Extract download URLs from the granules
    urls = []
    for granule in granules:
        for link in granule['links']:
            if link['rel'].endswith('/data#'):
                urls.append(link['href'])
                break
    
    # Test the first URL to check if data can be accessed using EosdisZarrStore
    test_response = requests.head(f'{urls[0]}.dmrpp')
    print('Can we use EosdisZarrStore and XArray to access these files?')
    print('Successful response from PO.DAAC' if test_response.ok else 'Could not get a response from PO.DAAC')
    #print("line 140", timeit.default_timer() - start_timeit)

    arrs_out = []
    # Loop through each dataset URL and process it
    for url in urls:
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Old code for pulling and saving each observation individually
        
        # Check if the file already exists, to avoid unnecessary downloads
        if os.path.isfile(f"{save_path}{name}.nc"):
            print(f"Some form of {save_path}{name}.nc already exists! Skipping for now...")
            continue
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        print(f"Pulling {url}")
    
        try:
            if not only_get_pixel_mask:
                # Select the "first" timestep to get rid of the time dimension for now.. is this really necessary?
                ds = xr.open_zarr(stores.EosdisStore(url), consolidated=False)[["quality_level","sea_surface_temperature"]].isel(time=0).load()
            elif only_get_pixel_mask:
                # Else only grab pixel mask
                ds = xr.open_zarr(stores.EosdisStore(url), consolidated=False)[["quality_level"]].isel(time=0).load()
            #print("line 165", timeit.default_timer() - start_timeit)      
            
            ds = ds.rename({"lat":"latitude","lon":"longitude"})
            #ds.to_netcdf("TEST_FULL_AVHRRF.nc")
            
            # We need to do an initial subset on VIIRS data, since the EosdisStore reader returns the 
            # entire swath if a portion of it falls in the target range.
            # For now I'm just subsetting +/- 2deg from the target point, but this can be refined.
            #print("line 173", timeit.default_timer() - start_timeit)
            ds_subset = swot_utils.xr_subset(ds,[lat-2, lat+2],[lon-2, lon+2])
            #print("line 175", timeit.default_timer() - start_timeit)
            
            # Skip datasets that are completely outside of target region
            if not isinstance(ds_subset, xr.Dataset):
                print("no valid data in bounds")
                continue
                
            if not only_get_pixel_mask and not filter:
                # If no filtering just pass the sst + quality_level dataset through..
                dataset_to_grid = ds_subset
                
            elif not only_get_pixel_mask and filter:
                # Apply quality control filtering on the SST (Sea Surface Temperature) variable
                sst_da = ds_subset.sea_surface_temperature
                quality_level_da = ds_subset.quality_level
                # Only retain SST values with quality level > 1, replacing others with NaN
                sst_da_q2 = sst_da.where(quality_level_da > quality_level, other=np.nan).rename("filtered_sea_surface_temperature")
                dataset_to_grid = xr.merge([sst_da,quality_level_da,sst_da_q2])
                
            elif not only_get_pixel_mask:
                # Apply quality control filtering on the SST (Sea Surface Temperature) variable
                sst_da = ds_subset.sea_surface_temperature
                quality_level_da = ds_subset.quality_level

                # Only retain SST values with quality level > 1, replacing others with NaN
                sst_da_q2 = sst_da.where(quality_level_da > quality_level, other=np.nan).rename("filtered_sea_surface_temperature")
                dataset_to_grid = xr.merge([sst_da,quality_level_da,sst_da_q2])

            elif only_get_pixel_mask:
                # Only use pixels with quality level above 1
                dataset_to_grid = ds_subset.quality_level.where(ds_subset.quality_level > 1,other=np.nan)
                
            #print("line 194", timeit.default_timer() - start_timeit)                
            # Interpolate to a standard grid
            # Obviously near the mask this may get a little weird...
            interp_sst = interp_utils.grid_everything(dataset_to_grid, lat, lon,  n=n, L_x=L_x, L_y=L_y)
            #print("line 198", timeit.default_timer() - start_timeit)              
            # Do some annoying xarray stuff to add the time dim
            interp_sst = interp_sst.expand_dims(dim={"time": 1},axis=0).assign_coords(time = ("time", [dataset_to_grid.time.values]))
            #print("line 201", timeit.default_timer() - start_timeit)   
            if only_get_pixel_mask:
                # Turn everything into booleans to cut down on memory
                interp_sst = interp_sst > 0
            
            '''
            if only_get_pixel_mask:
                # Save the mask as a boolean .npy file
                interp_sst.to_netcdf(f"{save_path}{save_name}")
            else:
                # Save the processed dataset as a NetCDF file in the specified directory
                interp_sst.to_netcdf(f"{save_path}{save_name}")
            '''
            
            arrs_out.append(interp_sst)
            
        except Exception as e:
            # Handle any errors that occur during processing
            print(e)
            traceback.print_exc()

    # Block for taking time mean
    ds_out = xr.concat(arrs_out,dim="time")
    if isinstance(timedelta_for_mean, type(None)):
        pass
    elif isinstance(timedelta_for_mean, np.timedelta64):
        if only_get_pixel_mask:
            # If we're trying to get the pixel mask take the 
            # maximum values (mask is either 0 or 1 at each timestep)
            ds_out = ds_out.resample(time='1D').max()
        else:
            # If we're trying to get average fields you should 
            # take the mean instead..
            ds_out = ds_out.resample(time='1D').mean()
    else:
        print("Timedelta_for_mean should be either Nonetype or np.timedelta64")

    #print("line 240", timeit.default_timer() - start_timeit)  
    ds_out.to_netcdf(f"{save_path}{name}")
    print(f"saved {save_path}{name}")
    #print("line 243", timeit.default_timer() - start_timeit)  

    return ds_out


# Function to download and subset VIIRS satellite data
def download_raw_SST(data_short_name, save_path, lat_min, lat_max, lon_min, lon_max, name,
                         start_time='2023-04-01T21:00:00Z', end_time='2023-07-28T20:59:59Z',
                         only_get_pixel_mask=False, quality_level=1):
    '''
    Downloads and subsets SST satellite data from NASA's CMR (Common Metadata Repository).
    The function searches for granules (data files) within a specified latitude/longitude range
    and filters the data based on quality control measures. Finally, the processed data is
    saved as a NetCDF file for further analysis.

    Most of the highres SST is from two infrared instrument types, the Advanced Very High 
    Resolution Radiometer (AVHRRF) and the Visible Infrared Imaging Radiometer Suite (VIIRS), 
    which have been used on multiple NOAA and European MetOp satellites. The AVHRRF data we 
    have access to appears to be from the MetOp-A, -B, and -C satellites (Oct 2007 - Nov 2021, 
    Sep 2012 - present, Nov 2018 - present) and VIIRS data from the NOAA S-NPP, NOAA-20, and 
    NOAA-2021 satellites (2011 - present, 2017 - present, 2022 - present).

    

    Parameters
    ----------
    data_short_name : str
        Short name of the dataset to be retrieved from NASA's CMR.
    save_path : str
        Directory path where the subsetted data files will be stored.
    lat : float
        Central latitude for the data search.
    lon : float
        Central longitude for the data search.
    name : str, optional
        A custom name tag to append to the saved file names.
    n : int, optional
        Grid size for interpolating the final dataset (default is 256).
    L_x : float, optional
        Grid resolution in the x-direction (default is 256 km, 
        corresponding to 1km x-resolution for n=256).
    L_y : float, optional
        Grid resolution in the y-direction (default is 256 km, 
        corresponding to 1km y-resolution for n=256).
    start_time : str, optional
        The UTC start time to bounding the search. Default is
        a little before the first day of the 1-day SWOT repeat phase.
    end_time : str, optional
        The UTC end time to bounding the search. Default is
        the a little after the last day of the 1-day SWOT repeat phase.
    timedelta_for_mean : numpy timedelta64, optional
        The time delta over which to calculate mean field values. If "None", skip
        averaging and return individual observations. Example 
    only_get_pixel_mask : bool, optional
        Boolean flag to instruct the script to (False) download sst, quality_flags,
        and filtered sst or (True) download a boolean pixel mask based on the "quality_level"
        parameter below. The latter option is good for saving memory if you just want the 
        pixel mask.
    quality_level : float, optional
        
        
    Returns
    -------
    None
        Saves subsetted data as either NetCDF (only_get_pixel_mask = False) 
        or .npy (only_get_pixel_mask = True) files in the save directory.
    '''
    #start_timeit = timeit.default_timer()
    # code you want to evaluate
    #elapsed = timeit.default_timer() - start_timeit
    
    # Define the NASA CMR API URL for searching granules
    cmr_url = 'https://cmr.earthdata.nasa.gov/search/granules.json'
    
    # Request granule metadata from the CMR API based on dataset name and search bounds
    print(lon_min,lat_min,lon_max,lat_max)
    
    response = requests.get(cmr_url, 
                            params={
                                'provider': DATA_PROVIDER,
                                'short_name': data_short_name, 
                                'temporal': f'{start_time},{end_time}',
                                'bounding_box': f'{lon_min},{lat_min},{lon_max},{lat_max}',
                                'page_size': 2000,
                                }
                           )
    #print("line 119", timeit.default_timer() - start_timeit)

    # Extract the list of granules from the response JSON
    try:
        granules = response.json()['feed']['entry']
    except:
        print(response.json())
    pprint(f"Found {len(granules)} granules!")
    if len(granules) < 1:
        print("No granules found, aborting...")
        return    
        
    # Extract download URLs from the granules
    urls = []
    for granule in granules:
        for link in granule['links']:
            if link['rel'].endswith('/data#'):
                urls.append(link['href'])
                break
    
    # Test the first URL to check if data can be accessed using EosdisZarrStore
    test_response = requests.head(f'{urls[0]}.dmrpp')
    print('Can we use EosdisZarrStore and XArray to access these files?')
    print('Successful response from PO.DAAC' if test_response.ok else 'Could not get a response from PO.DAAC')
    #print("line 140", timeit.default_timer() - start_timeit)

    arrs_out = []
    # Loop through each dataset URL and process it
    for url in urls:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Old code for pulling and saving each observation individually
        save_name = f"{(url.split("/")[-1]).split(".")[0]}_{name}.nc"        
        # Check if the file already exists, to avoid unnecessary downloads
        if os.path.isfile(f"{save_path}{save_name}"):
            print(f"Some form of {save_path}{save_name} already exists! Skipping for now...")
            continue
        # Create the save path if it doesn't already exist
        elif not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=False)
        
        print(f"Pulling {url}")
        try:
            if not only_get_pixel_mask:
                # Select the "first" timestep to get rid of the time dimension for now.. is this really necessary?
                ds = xr.open_zarr(stores.EosdisStore(url), consolidated=False)[["quality_level","sea_surface_temperature"]].isel(time=0).load()
            elif only_get_pixel_mask:
                # Else only grab pixel mask
                ds = xr.open_zarr(stores.EosdisStore(url), consolidated=False)[["quality_level"]].isel(time=0).load()
            #print("line 165", timeit.default_timer() - start_timeit)      
            
            ds = ds.rename({"lat":"latitude","lon":"longitude"})

            print(f"Saving {save_name} to {save_path}")
            
            ds.to_netcdf(f"{save_path}{save_name}")
            
        except Exception as e:
            # Handle any errors that occur during processing
            print(e)
            traceback.print_exc()

    return 

"""