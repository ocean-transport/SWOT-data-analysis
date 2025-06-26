#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains scripts for downloading SWOT data from the CNES AVISO
THREDDS Data Server. In addition to the listed Python dependencies, you will
need to include your own AVISO+ credetials to retrieve SWOT data from AVISO:
https://www.aviso.altimetry.fr/en/my-aviso-plus.html.



Functions:
1. list_nc_files_from_thredds_catalog: 
2. download_nc_file: 
3. run_download: 

Author: Jinbo Wang (First version), Tatsu
Date: First version: 11.15.2024

Dependencies:
    - numpy
    - xarray
    - paramiko
    - geopandas
    - shapely
    - tqdm 
    - xml
"""

import os
import argparse
import warnings
import requests
import xml.etree.ElementTree as ET # For parsing the THREDDS 
import tqdm
import geopandas as gpd  # For working with geospatial data
import shapely.geometry as geometry  # For defining bounding boxes and geometry
from itertools import groupby

import sys
sys.path.append('../../SWOT-data-analysis/src')
import download_swaths

# Turn off SSL warnings (optional)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# XML Catalog Parsing
# ─────────────────────────────────────────────
def list_nc_files_from_thredds_catalog(catalog_url, ssh_kwargs):
    """Return list of .nc filenames from a THREDDS catalog URL."""
    response = requests.get(catalog_url, auth=(ssh_kwargs["username"], ssh_kwargs["password"]))
    response.raise_for_status()

    ns = {'ns': 'http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0'}
    root = ET.fromstring(response.content)

    datasets = root.findall('.//ns:dataset', namespaces=ns)
    return [ds.attrib['name'] for ds in datasets if 'name' in ds.attrib and ds.attrib['name'].endswith('.nc')]

# ─────────────────────────────────────────────
# File Downloader with Progress Bar
# ─────────────────────────────────────────────
def download_nc_file(download_url, save_path, ssh_kwargs):
    """Download a .nc file from THREDDS fileServer with a progress bar, if not already downloaded."""
    os.makedirs(save_path, exist_ok=True)
    local_filename = os.path.join(save_path, os.path.basename(download_url))

    if os.path.exists(local_filename):
        print(f"✔ File already exists, skipping: {local_filename}")
        return local_filename

    with requests.get(download_url, stream=True, auth=(ssh_kwargs["username"], ssh_kwargs["password"])) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        chunk_size = 512 * 1024  # 512 KB

        with open(local_filename, 'wb') as f, tqdm.tqdm(
            total=total_size, unit='B', unit_scale=True, desc=os.path.basename(local_filename)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return local_filename


# ─────────────────────────────────────────────
# Find swaths within a specified lat-lon range
# ─────────────────────────────────────────────
def find_swaths(sw_corner, ne_corner, path_to_sph_file="./orbit_data/sph_science_swath.zip"):
    """
    Finds SWOT passes that intersect a given geographical bounding box.
    - Uses GeoPandas to load the shapefile and filter passes based on intersection with the bounding box.
    - The bounding box is extended by 0.2° to account for potential nadir data overlaps.
    """
    try:
        # Load the SWOT shapefile containing orbital data
        gdf_karin = gpd.read_file(path_to_sph_file)
    except:
        print("Error: Ensure the shapefile exists in the correct directory.")
        return []

    # Define the bounding box as a Shapely geometry
    bbox = geometry.box(sw_corner[0], sw_corner[1], ne_corner[0], ne_corner[1])

    # Extend the bounding box by 0.2° for better coverage
    extended_bbox = geometry.box(
        bbox.bounds[0] - 0.2, bbox.bounds[1] - 0.2,
        bbox.bounds[2] + 0.2, bbox.bounds[3] + 0.2
    )

    # Filter the GeoDataFrame for passes that intersect the bounding box
    overlapping_segments = gdf_karin[gdf_karin.intersects(extended_bbox)]

    # Extract pass IDs and format them as 3-digit strings with leading zeros
    pass_IDs_list = []
    for foo in overlapping_segments["ID_PASS"].values:
        pass_IDs_list.append(str(foo).zfill(3))

    return pass_IDs_list



# ─────────────────────────────────────────────
# Main Logic to Download SWOT .nc Files
# ─────────────────────────────────────────────
def run_download(sw_corner, ne_corner, cycles, remote_path, save_path, orbit_file_path, ssh_kwargs):
    """
    Download SWOT .nc files from THREDDS that intersect with a region
    and match given cycles.
    """
    # Specify the remote and catalog paths.
    # Add some logic in case people speciify the entire remote path..
    if "https://tds-odatis.aviso.altimetry.fr/" in remote_path:
        remote_path = "dataset"+remote_path.split("dataset")[1]
    if "catalog.html" in remote_path:
        remote_path = remote_path.split("catalog.html")[0]
    remote_catalog_path = f"https://tds-odatis.aviso.altimetry.fr/thredds/catalog/{remote_path}"
    remote_fileserver_path = f"https://tds-odatis.aviso.altimetry.fr/thredds/fileServer/{remote_path}"
    
    # Split the cycles into calval and science
    cycle_split = []
    for k, g in groupby(sorted(cycles), lambda x: int(x) >= 472):
        cycle_split.append(list(g))
    # Get Pass IDs for Science cycles
    for cycle in cycle_split[0]:
        pass_IDs_list = find_swaths(sw_corner, ne_corner, path_to_sph_file=f"{orbit_file_path}/sph_science_nadir.zip")
    science_cycle_n = len(pass_IDs_list)
    print(f"Found {len(pass_IDs_list)} passes in {len(cycle_split[0])} cycles for the science phase.")
    # Get Pass IDs for CalVal cycles
    for cycle in cycle_split[1]:
        pass_IDs_list += find_swaths(sw_corner, ne_corner, path_to_sph_file=f"{orbit_file_path}/shp_calval_nadir.zip")
    print(f"Found {len(pass_IDs_list)-science_cycle_n} passes in {len(cycle_split[1])} cycles for the calval phase")

    # Loop through cycles
    for cycle in cycles:
        cycle_str = str(cycle).zfill(3)
        catalog_url = f"{remote_catalog_path}/cycle_{cycle_str}/catalog.xml"
        print(f"Fetching catalog for cycle {cycle_str}...")

        try:
            nc_files = list_nc_files_from_thredds_catalog(catalog_url, ssh_kwargs)
        except Exception as e:
            print(f"Failed to fetch or parse catalog for cycle {cycle_str}: {e}")
            continue

        product = remote_path.split("/")[-1]
        for pass_id in pass_IDs_list:
            for nc_file in nc_files:
                if pass_id in nc_file.split(f"{product}_{cycle_str}_")[1].split("_")[1]:
                    download_url = f"{remote_fileserver_path}/cycle_{cycle_str}/{nc_file}"
                    save_path = os.path.join(save_path, f"cycle_{cycle_str}")
                    try:
                        download_nc_file(download_url, save_path, ssh_kwargs)
                    except Exception as e:
                        print(f"Failed to download {download_url}: {e}")    



# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SWOT .nc files for a given region and cycles.")
    
    parser.add_argument("--sw_corner", nargs=2, type=float, required=True, help="Southwest corner (lon lat)")
    parser.add_argument("--ne_corner", nargs=2, type=float, required=True, help="Northeast corner (lon lat)")
    parser.add_argument("--cycles", nargs='+', type=int, required=True, help="Cycle number to download")

    # Specify remote path to access data and local path to save data
    parser.add_argument("--remote_path", nargs=1, type=string, required=True, help="Path to access remote files. NOTE: this path needs to includ \
                                                                                    both the overall version (e.x. 'dataset-l3-swot-karin-nadir-validated'), the  \
                                                                                    processing version (e.x. 'l3_lr_ssh'), and the base product version (e.x. \
                                                                                    'Unsmoothed'). You can look at the data products available by accessing the \
                                                                                    THREDDS catalog at https://tds-odatis.aviso.altimetry.fr/thredds/catalog/ " )
    parser.add_argument("--save_path", nargs=1, type=string, required=True, help="Path to save downloaded files.")

    # Optional arguments
    parser.add_argument("--orbit_file_path", type=string, required=False, default="../orbit_data", help="Path to sph orbit files. Defualt is ../orbit_data/ directory")    
    parser.add_argument("--username", nargs=1, type=string, required=False, default="tdmonkman@uchicago.edu", help="AVISO+ username")
    parser.add_argument("--password", nargs=1, type=string, required=False, default="2prSvl", help="AVISO+ passowrd")
    
    args = parser.parse_args()
    
    ssh_kwargs = {"username":args.username,"password":args.password}
    run_download(args.sw_corner, args.ne_corner, args.cycles, args.remote_path, args.save_path, args.orbit_file_path, ssh_kwargs)


