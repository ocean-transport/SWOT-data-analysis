import xarray as xr
import numpy as np

# Other Python libraries
import requests

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
import tatsu_swot_utils as swot_utils
from importlib import reload
import os
import traceback

# For sbatch script
import argparse


stores = reload(stores)
dmrpp = reload(dmrpp)
swot_utils = reload(swot_utils)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Global variables
DATA_PROVIDER = 'POCLOUD'


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Download and subset VIIRS data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def download_subset_VIIRS(data_short_name,save_path, name="", n=256, L_x=256e3, L_y=256e3):

    # Bounding box around the California region
    lats = slice(30, 40)
    lons = slice(-130, -120)

    # Start stop times
    start_time = '2023-04-01T21:00:00Z'
    end_time = '2023-07-28T20:59:59Z'
    
    cmr_url = 'https://cmr.earthdata.nasa.gov/search/granules.json'
    
    response = requests.get(cmr_url, 
                            params={
                                'provider': DATA_PROVIDER,
                                'short_name': data_short_name, 
                                'temporal': f'{start_time},{end_time}',
                                'bounding_box': f'{lons.start},{lats.start},{lons.stop},{lats.stop}',
                                'page_size': 2000,
                                }
                           )

    granules = response.json()['feed']['entry']
    pprint(f"Found {len(granules)} granules!")
    
    urls = []
    for granule in granules:
        for link in granule['links']:
            if link['rel'].endswith('/data#'):
                urls.append(link['href'])
                break

    test_response = requests.head(f'{urls[0]}.dmrpp')
    print('Can we use EosdisZarrStore and XArray to access these files?')
    print('Successful response from PO.DAAC' if test_response.ok else 'Could not get a response from PO.DAAC')


    for url in urls:
        # Define the save file name / path 
        save_name = f"{(url.split("/")[-1]).split(".")[0]}_{name}.nc"
        
        if os.path.isfile(f"{save_path}{save_name}"):
            # Check that you aren't redownloading a file that already exists
            print(f"Some form of {save_path}{save_name} already exists! Skipping for now...")
            continue
            
        print(f"Pulling {url}")
        

        try:
            ds = xr.open_zarr(stores.EosdisStore(url), consolidated=False)[["quality_level","sea_surface_temperature"]].isel(time=0)
            ds = ds.rename({"lat":"latitude","lon":"longitude"})
            
            # We need to do an initial subset on VIIRS data, since the EosdisStore reader returns the 
            # entire swath if a portion of it falls in the target range.
            ds_subset = swot_utils.subset(ds,[20,50],[-140, -110])
    
            # Skip datasets that are completely outside of target region
            if type(ds_subset) is None:
                print("no valid data in bounds")
                continue
    
            # Use the quality level flag to filter the SST 
            sst_da = ds_subset.sea_surface_temperature
            quality_level_da = ds_subset.quality_level
    
            # Filter the SST
            sst_da = sst_da.where(quality_level_da>4, other=np.nan)
            # Interpolate to a standard grid
            interp_sst = interp_utils.grid_everything(sst_da, 36.46, -125.74,  n=n, L_x=L_x, L_y=L_y)
            # Do some annoying xarray stuff to add the time dim
            interp_sst = interp_sst.expand_dims(dim={"time": 1},axis=0).assign_coords(time = ("time", [sst_da.time.values]))
            
            # save to a local directory for now...
            interp_sst.to_netcdf(f"{save_path}{save_name}")
            ds.close()
            
        except Exception as e:
            print(e)
            traceback.print_exc()
        
    return




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# virrs_snpp_short_name = 'VIIRS_NPP-STAR-L2P-v2.80'
# virrs_noaa20_short_name = 'VIIRS_N20-STAR-L2P-v2.80'
# virrs_noaa21_short_name = 'N21-VIIRS-L2P-ACSPO-v2.80'
# save_path = "../test_VIIRS_subsetting/"


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
parser.add_argument("save_name")
parser.add_argument("n")
parser.add_argument("L_x")
parser.add_argument("L_y")

args = parser.parse_args()

download_subset_VIIRS(str(args.data_short_name),
                      str(args.save_path),
                      str(args.save_name),
                      int(args.n),
                      int(args.L_x),
                      int(args.L_y),
                     )



