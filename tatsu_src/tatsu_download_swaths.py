#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains routines for manipulating SWOT SSH data.
Based on the Jinbo Wang's SWOT-OpenToolkit code

1st Author: Jinbo Wang
2nd Author: Tatsu
Date: First version: 11.15.2024

Dependencies:
    numpy
    scipy
    pylab
    xarray
    pyresample
    json
    s3fs
    requests
    h5netcdf
"""

import geopandas as gpd
import shapely.geometry as geometry
import paramiko
import os
import xarray as xr

# Add the path to the Tatsu's swot library
import sys
sys.path.append('./tatsu_src/')
import tatsu_swot_utils as tatsu_swot


#*********************************************************************
def find_swaths(sw_corner,ne_corner,path_to_sph_file="./orbit_data/sph_science_swath.zip"):
    """
    Script based on Jinbo's code on the SWOT-OpenToolkit. This script uses the orbit shapefiles
    stored in the "orbit_data" directory. 

    Variables
    -----
    sw_corner:        list, list of 
    ne_corner:        list, 
    path_to_sph_file: string,
    
    Output
    -----
    pass_IDs_list:    list,
    
    """
    
    # Load the shapefile
    try:
        gdf_karin = gpd.read_file(path_to_sph_file)
    except:
        print("Make sure you've downloaded the shape \
               file and put it in the correct directory!")
    # define the bounding box
    bbox = geometry.box(sw_corner[0], sw_corner[1], ne_corner[0], ne_corner[1])
    # extend the bounding box by 0.2 degree
    extended_bbox = geometry.box(bbox.bounds[0] - 0.2, bbox.bounds[1] - 0.2, bbox.bounds[2] + 0.2, bbox.bounds[3] + 0.2) #for nadir data
    
    # Filter the GeoDataFrame for rows that intersect with the extended bounding box
    overlapping_segments = gdf_karin[gdf_karin.intersects(bbox)]
    
    # Pull pass numbers that we are interested in
    pass_IDs_list = []
    for foo in overlapping_segments["ID_PASS"].values:
        foo_str = str(foo)
        # Add leading zeros to pass number
        foo_str = foo_str.zfill(3)
        pass_IDs_list.append(foo_str)

    return pass_IDs_list
#*********************************************************************


#*********************************************************************
def download_passes(pass_ID, cycle="001",remote_path="swot_products/l3_karin_nadir/l3_lr_ssh/v1_0_2/Unsmoothed",
                    save_path=f"../../SWOT_L3/Unsmoothed/cycle_001", \
                    hostname = "ftp-access.aviso.altimetry.fr", port = 2221, \
                    username = "tdmonkman@uchicago.edu", password = "2prSvl", \
                    subset=False, lat_lims=False, trim_suffix="trimmed"): 
    """
    Script for downloading specific passes off of AVISO+ using sftp. This script will
    establish a connection to the aviso ftp server using secure shell (the "s" in "sftp").
    The specific aviso ftp server you are pulling data from is specified in the "hostname" 
    variable. The path to the SWOT data on the aviso ftp server is given by "remote_path."
    If you would like to change the version/release of the SWOT you will need to edit the 
    remote path variable, assuming the data is on aviso. If you are changing the data release 
    it would be good to edit the save path as well.

    There are helper scripts to load in the cycle and specific product. See the DEMO at the 
    bottom of this file.
    
    Some other notes:
    I'm establishing a new sftp connection for each file, in case of random
    network issues. I've included use my username and password to make things smoother.

    This script has some functionality to subset swaths by latitude and trim off the
    northern / southern ends. I added a suffix, "trim_suffix," to add to the saved 
    swath file name to specify that the swath has been modified.
    
    Variables
    -----
    pass_ID:       string, numeric ID of specific pass you would like to download. 
    cycle:         string, numeric ID of the cycle you would like to download. Expected to 
                           contain 3 digits with leading zeros i.e. ("001", "002",...,"140")
    remote_path:   string, path to SWOT data on aviso
    save_path:     string, path to save files locally. I put my data a couple of levels up
    hostname:      string, url to aviso ftp server
    port:          int, network port to connect to on remote server
    username:      string, aviso username
    password:      string, aviso password
    subset:        bool, If True, subset swath to only include data between the values specified in 
                         "lat_lims". If False (default), download the entire swath
    lat_lims:      list, list of integers specifying bounding latitudes of the swath when trimming
    trim_suffix:   string, string to add to trimmed files to show that they are trimmed.
    
    Output
    -----
    Data is saved to "save_path"
    
    """
    # Get the SWOT release version you are trying to pull THIS IS HACKY FIX THIS
    version = remote_path.split("_lr_ssh")[0][-1]
    try:
        version = int(version)
    except:
        print(f"Invalid input: SWOT release L{verison} in {remote_path}")
        print(f"Are you sure you inputed the right file path?")
        return
    # Roughly define target remote file (don't know the datetime part of the filename)
    if "Unsmoothed" in remote_path:
        target_remote_file = f"SWOT_L{version}_LR_SSH_Unsmoothed_{cycle}_{pass_ID}"
    elif "Expert" in remote_path:
        target_remote_file = f"SWOT_L{version}_LR_SSH_Expert_{cycle}_{pass_ID}"
        
    # Create an SSH client
    print("Attempting SSH connection...")
    
    # Open SSH client
    with paramiko.SSHClient() as ssh_client: 
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname, port, username, password,timeout=30)
       
        # Create sftp client
        sftp_client = ssh_client.open_sftp()
        
        # Define path to the specific cycle you want to pull from the remote server 
        remote_path_cycle = f"{remote_path}/cycle_"+cycle

        # Check whether the path exists
        try:
            sftp_client.stat(remote_path_cycle)
            print(f"Found cycle_{cycle} on remote server")
        except paramiko.SSHException:
            print(f"Can't find path for cycle_{cycle} on remote server")
        
        # List files on sftp server
        remote_files = sftp_client.listdir(remote_path_cycle)

        # Look for a match on the remote server
        print(f"Looking for matches to {target_remote_file}...")
        for remote_file in remote_files:
            # Loop through each file in list, continue if not a match
            if not target_remote_file in remote_file:
                continue
            # If you find the file make sure to print
            print(f"Found remote file {remote_file}")
            # Make sure the file isn't downloaded already
            if os.path.isfile(f"./{save_path}/{remote_file}"):
                print(f"./{save_path}/{remote_file} already exists!")
                continue
            # Else, try and download it
            try:
                # If you aren't subsetting just download the whole file (~400Mb)
                if not subset:
                        sftp_client.get(f"{remote_path_cycle}/{remote_file}", f"./{save_path}/{remote_file}")
                        print(f"Downloaded full {remote_file}")
                elif subset:
                    # First check that the trimmed file doesn't already exist
                    trimmed_filename = f"{remote_file[:-3]}_{trim_suffix}.nc"
                    if os.path.isfile(f"./{save_path}/{trimmed_filename}"):
                        print(f"./{save_path}/{trimmed_filename} already exists!")
                        continue
                    print(f"Subsetting {remote_file}")
                    # If you are subsetting you still need to download the 
                    # whole file since we're using sftp :(
                    # I'm going to save a tmp file to the cwd and then delete it..
                    sftp_client.get(f"{remote_path_cycle}/{remote_file}", f"./tmp{trim_suffix}.nc")
                    swath = xr.open_dataset(f"./tmp{trim_suffix}.nc")
                    # Trim the swath using Jinbo's script
                    trimmed_swath = tatsu_swot.subset(swath,lat_lims).load()
                    # Add some attributes to the dataset to keep track of things
                    trimmed_swath.assign_attrs(cycle=cycle,
                                               pass_ID=pass_ID,
                                               swath_name=remote_file
                                              )
                    # Rename the file, I'm assuming it's just a .nc file
                    trimmed_swath.to_netcdf(f"./{save_path}/{trimmed_filename}")   
                    # Remove the temporary file
                    os.remove(f"tmp{trim_suffix}.nc")
            except Exception as e:
                # Code to handle the exception
                print(f"Failed to download {remote_file}")
                print(f"An error occured:", e)
                # If the download was unsuccessful due to a timeout 
                # you may need to delete a partially downloaded incomplete file in the target directory
                if subset and os.path.isfile(f"./{save_path}/{trimmed_filename}"):
                    os.remove(f"./{save_path}/{trimmed_filename}")
                elif os.path.isfile(f"./{save_path}/{trimmed_filename}"):
                    os.remove(f"./{save_path}/{remote_file}")
                # Record that the file failed to download
                with open("skipped_swaths.txt","a") as file:
                    file.write(f"\n {cycle}, {pass_ID}")
            else:
                pass

    return
#*********************************************************************


def clean_incomplete_files(path, size=10): 
    """
    Simple script to delete any incomplete files in a target  directory.
    For example this may be useful if there were timeouts when trying to download
    a set of swaths, leaving a bunch of incomplete files on the download directory

    Variables
    -----
    path: string, Path to location of incomplete files 
    size: int or flt, Minimum size of file in Mb 
    
    Outputs
    -----
    
    
    """
    files = os.listdir(path)

    for file in files:
        # Find size of file, 
        file_stats = os.stat(f"{path}/{file}")
        if file_stats.st_size / (1024 * 1024) < size:
            os.remove(f"{path}/{file}")
            print(f"Deleted possible incomplete file {path}/{file}")


    return







'''
#*********************************************************************
# DEMO
#*********************************************************************
import os
import time

# Add the path to the Tatsu's swot download library
import sys
sys.path.append('./tatsu_src/')
import tatsu_download_swaths as tatsu_download

#turn off warnings
import warnings
warnings.filterwarnings("ignore")

# Define connection parameters
ssh_kwargs = {
              "hostname":"ftp-access.aviso.altimetry.fr",
              "port": 2221,
              "username":"tdmonkman@uchicago.edu",
              "password":"2prSvl"
            }

# Define bounding domain
# small Agulhas
# sw_corner = [10.0,  -42.0] # [degE, degN]
# ne_corner = [18.0, -36.0] # [degE, degN]
# big Agulhas-50,-30
sw_corner = [5.0, -50.0]
ne_corner = [30.0, -30.0]
# West Greenland 
# sw_corner = [-53.55, 71.32]
# ne_corner = [-51.33, 72.0]
# Global dataset
# sw_corner = [-180, -90]
# ne_corner = [180, 90]

lat_lims = [sw_corner[1],ne_corner[1]]

# Specify cycles you want to downnload
# Cycles 001 - 016 are for the science orbit
# cycles = [str(c_num).zfill(3) for c_num in range(1,17)]
# Cycles 474 - 578 are from the 1-day repeat 
cycles = [str(c_num).zfill(3) for c_num in range(525,579)]

# Use sph_science_swath for the 21-day repeat
# path_to_sph_file="../orbit_data/sph_science_swath.zip"
# Use sph_calval_swath for the 1-day repeats
path_to_sph_file="../orbit_data/sph_calval_swath.zip"

# Get pass IDs for swaths that intersect your box
pass_IDs_list = tatsu_download.find_swaths(sw_corner, ne_corner,
                                           path_to_sph_file=path_to_sph_file)


# pass_IDs_list is just a list of 3-digit strings with the pass number and 
# leading zeros, i.e. ["001","555",etc...]. You can write your own if 
# you know which swaths you want.
# For example pass_IDs_list = ["001","013","016","026"] gives you SoCal and Agulhas
pass_IDs_list = ["001","013","016","026"]

# Paths for L3 unsmoothed data
remote_path="swot_products/l3_karin_nadir/l3_lr_ssh/v1_0_2/Unsmoothed"
local_path = "../../SWOT_L3/Unsmoothed"

# Paths for L2 expert data
# remote_path="/swot_products/l2_karin/l2_lr_ssh/PIC0/Expert"
# local_path = "../../SWOT_L3/Unsmoothed"


# Make a file to store IDs for swaths that didn't download
with open("skipped_swaths.txt","w") as file:
    file.write("Failed to download the following swaths:")
    file.write(" cycle, pass_ID \n ----------")
    file.close()

# MAKE SURE TO CHANGE THE SAVE PATH
for cycle in cycles:
    save_path = local_path+f"/cycle_{cycle}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=False)
    # If you want to clean any existing files in the path,
    # do so here.
    tatsu_download.clean_incomplete_files(save_path, size=10)
    
    # Download passes
    for pass_ID in pass_IDs_list:
        try:
            tatsu_download.download_passes(pass_ID,cycle=cycle,remote_path=remote_path,
                                           save_path=save_path,**ssh_kwargs,
                                           subset=False,lat_lims=lat_lims,trim_suffix="")
        except Exception as e:
            print("*****"*5)
            print(f"Could not download pass {pass_ID} in cycle {cycle}")
            print(f"An error occured: {e}")
            print("*****"*5)
            with open("skipped_swaths.txt","a") as file:
                file.write(f"\n {cycle}, {pass_ID}")
                
    # Sleep for 10 seconds so you don't make AVISO mad 
    time.sleep(10)

    
'''
