import xarray as xr
import numpy as np
from glob import glob

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
sys.path.append('../zarr-eosdis-store1/eosdis_store1/')
sys.path.append('../../SWOT-data-analysis/src')
import stores
import dmrpp
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

# Function to download and subset VIIRS satellite data
def download_subset_grid_SST(data_short_name, save_path, lat, lon, name="", n=256, L_x=256e3, L_y=256e3,
                         start_time='2023-04-01T21:00:00Z', end_time='2023-07-28T20:59:59Z',
                         timedelta_for_mean=None, only_get_pixel_mask=False, quality_level=1):
    """
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
    """
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

            if not only_get_pixel_mask:
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
            
            """
            if only_get_pixel_mask:
                # Save the mask as a boolean .npy file
                interp_sst.to_netcdf(f"{save_path}{save_name}")
            else:
                # Save the processed dataset as a NetCDF file in the specified directory
                interp_sst.to_netcdf(f"{save_path}{save_name}")
            """
            
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
    print("{save_path}{name}")
    ds_out.to_netcdf(f"{save_path}{name}")
    #print("line 243", timeit.default_timer() - start_timeit)  

    return ds_out


# Function to download and subset VIIRS satellite data
def download_raw_SST(data_short_name, save_path, lat_min, lat_max, lon_min, lon_max, name,
                         start_time='2023-04-01T21:00:00Z', end_time='2023-07-28T20:59:59Z',
                         only_get_pixel_mask=False, quality_level=1):
    """
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
    """
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



def subset_raw_SST(file_path, lat, lon, save_name="", n=256, L_x=256e3, L_y=256e3, subset_deg=3,
                     timedelta_for_mean=None, only_get_pixel_mask=False, quality_level=1):
    """
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
    file_path : str
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
    """

    files = glob(f"{file_path}/*.nc")
    
    
    # Loop through each dataset URL and process it
    for file in files:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Check if the file already exists, to avoid unnecessary downloads
        if os.path.isfile(f"{save_path}{save_name}"):
            print(f"Some form of {save_path}{save_name}.nc already exists! Skipping for now...")
            continue
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        try:
            if "lat" in ds.dims:
                ds = ds.rename({"lat":"latitude","lon":"longitude"})
            
            # We need to do an initial subset on VIIRS data, since the EosdisStore reader returns the 
            # entire swath if a portion of it falls in the target range.
            # For now I'm just subsetting +/- 3deg from the target point, but this can be refined.
            ds_subset = swot_utils.xr_subset(ds,
                                             [lat-subset_deg, lat+subset_deg],
                                             [lon-subset_deg, lon+subset_deg]
                                            )
            
            # Skip datasets that are completely outside of target region
            if not isinstance(ds_subset, xr.Dataset):
                print("no valid data in bounds")
                continue

            if not only_get_pixel_mask:
                # Apply quality control filtering on the SST (Sea Surface Temperature) variable
                # This assumes that we've download SST fields along with the quality level, which is 
                # not guarunteed. 
                sst_da = ds_subset.sea_surface_temperature
                quality_level_da = ds_subset.quality_level

                # Only retain SST values with quality level > 1, replacing others with NaN
                sst_da_q2 = sst_da.where(quality_level_da > quality_level, other=np.nan).rename("filtered_sea_surface_temperature")
                dataset_to_grid = xr.merge([sst_da,quality_level_da,sst_da_q2])
           
            elif only_get_pixel_mask:
                # Only use pixels with quality level above 1
                dataset_to_grid = ds_subset.quality_level.where(ds_subset.quality_level > 1,other=np.nan)
                              
            # Interpolate to a standard grid
            # Obviously near the mask this may get a little weird...
            interp_sst = interp_utils.grid_everything(dataset_to_grid, lat, lon,  n=n, L_x=L_x, L_y=L_y)            
            # Do some annoying xarray stuff to add the time dim
            interp_sst = interp_sst.expand_dims(dim={"time": 1},axis=0).assign_coords(time = ("time", [dataset_to_grid.time.values]))
            if only_get_pixel_mask:
                # Turn everything into booleans to cut down on memory
                interp_sst = interp_sst > 0
            
            """
            if only_get_pixel_mask:
                # Save the mask as a boolean .npy file
                interp_sst.to_netcdf(f"{save_path}{save_name}")
            else:
                # Save the processed dataset as a NetCDF file in the specified directory
                interp_sst.to_netcdf(f"{save_path}{save_name}")
            """
            
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
    print("{save_path}{name}")
    ds_out.to_netcdf(f"{save_path}{save_name}")
    #print("line 243", timeit.default_timer() - start_timeit)  

    return ds_out
    



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