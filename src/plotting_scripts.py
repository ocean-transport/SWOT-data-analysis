#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains routines for plotting SWOT (Surface Water and Ocean Topography) data.
It includes utilities to load bathymetry data, remap quality flags, and plot SWOT swaths with geographic features.

Author: Tatsu
Date: First version: 1.23.2025

Dependencies:
    - xarray
    - numpy
    - matplotlib
    - cartopy
    - cmocean
"""

# Import necessary libraries
import xarray as xr  # For working with multidimensional SWOT data
import os  # For file path and directory management
import numpy as np  # For numerical operations
from glob import glob  # For finding files with patterns

# Import visualization libraries
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs  # For geographic projections
import cartopy.feature as cfeature  # For adding features (e.g., coastlines)
import cartopy.io.shapereader as shpreader  # For reading shapefiles (e.g., bathymetry data)

import cmocean.cm as cm  # Oceanographic colormaps (e.g., "balance")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Function: load_bathymetry
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_bathymetry(zip_file_url):
    """
    Loads bathymetry data from a Natural Earth zip file containing shapefiles.

    Parameters
    ----------
    zip_file_url : str
        URL of the zip file containing bathymetry shapefiles.

    Returns
    -------
    depths : np.ndarray
        Sorted array of depth levels (from surface to bottom).
    shp_dict : dict
        Dictionary mapping depth levels (as strings) to their respective shapefile geometries.
    """
    # Import required libraries for downloading and extracting zip files
    import io
    import zipfile
    import requests

    # Check if bathymetry data is already downloaded
    if not os.path.exists("../ne_10m_bathymetry_all/"):
        print("Downloading bathymetry shapefiles...")
        # Download the zip file and extract its contents
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("../ne_10m_bathymetry_all/")  # Extract files to the specified directory

    # Read all shapefiles in the extracted directory
    shp_dict = {}
    files = glob('../ne_10m_bathymetry_all/*.shp')  # Get all shapefiles
    assert len(files) > 0, "No shapefiles found in the directory!"
    files.sort()  # Sort the files to process them in order

    # Extract depth levels from the filenames and load the geometries
    depths = []
    for f in files:
        # Extract depth value from the filename (e.g., "-2000" from "ne_10m_bathymetry_-2000.shp")
        depth = '-' + f.split('_')[-1].split('.')[0]
        depths.append(depth)

        # Load the shapefile using Cartopy's `shpreader`
        bbox = (-180, -90, 180, 90)  # Global bounding box (lon_min, lat_min, lon_max, lat_max)
        nei = shpreader.Reader(f, bbox=bbox)
        shp_dict[depth] = nei

    # Return depths (sorted from surface to bottom) and the shapefile dictionary
    depths = np.array(depths)[::-1]  # Reverse to get surface-to-bottom order
    return depths, shp_dict


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Mini script for testing bathymetry loading, may delete
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == "__main__":
    # Load bathymetry shapefiles from Natural Earth repository
    depths_str, shp_dict = load_bathymetry(
        'https://naturalearth.s3.amazonaws.com/' +
        '10m_physical/ne_10m_bathymetry_all.zip'
    )

    # Create a colormap for bathymetry depth levels
    depths = depths_str.astype(int)  # Convert depth strings to integers
    N = len(depths)  # Number of depth levels
    nudge = 0.01  # Slight adjustment to bin edges
    boundaries = [min(depths)] + sorted(depths + nudge)  # Bin edges for colormap
    norm = matplotlib.colors.BoundaryNorm(boundaries, N)  # Normalize depth values
    blues_cm = matplotlib.colormaps['Blues_r'].resampled(N)  # Use reversed "Blues" colormap
    colors_depths = blues_cm(norm(depths))  # Map depth values to corresponding colors
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Function: remap_quality_flags
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def remap_quality_flags(swath):
    """
    Remaps quality flags in the SWOT dataset to discrete values for easier plotting.

    Parameters
    ----------
    swath : xarray.Dataset
        SWOT dataset containing the 'quality_flag' variable.

    Returns
    -------
    xarray.Dataset
        Modified dataset with remapped 'quality_flag' values.
    """
    # Check if 'quality_flag' exists in the dataset
    if not "quality_flag" in swath:
        return

    # Replace original quality flag values with simplified discrete values
    flags = swath.quality_flag
    flags.values[flags.values == 5.] = 1
    flags.values[flags.values == 10.] = 2
    flags.values[flags.values == 20.] = 3
    flags.values[flags.values == 30.] = 4
    flags.values[flags.values == 50.] = 5
    flags.values[flags.values == 70.] = 6
    flags.values[flags.values == 100.] = 7
    flags.values[flags.values == 101.] = 8
    flags.values[flags.values == 102.] = 9

    # Update the dataset and return it
    swath.quality_flag.values = flags.values
    return swath


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Function: plot_cycle
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_cycle(swaths,fields=["ssha"],title="Example Swaths August 2024 (Cycle 4)",cbar_titles=["SSHA (m)"],
               vmins=[-0.8],vmaxes=[0.8], subplot_kw={'projection': ccrs.PlateCarree()},
               ssha_plot_kw = {"transform":ccrs.PlateCarree(),
                              "s":1,"marker":".","alpha":1,"linewidths":0},
               cmaps=[cm.balance],dpi=100,
               set_extent=False,extent_lims=[-180,180,-90,90],
               axes=[None], plot_bathymetry=True, add_labels=False,
              ):
    """
    Plots SWOT swaths for a single cycle with optional bathymetry and other geographic features.

    Parameters
    ----------
    swaths : list of xarray.Dataset
        List of SWOT swaths to plot.
    fields : list of str, optional
        Data variables to plot (default: ["ssha"]).
    title : str, optional
        Title of the plot (default: "Example Swaths August 2024 (Cycle 4)").
    cbar_titles : list of str, optional
        Titles for colorbars (default: ["SSHA (m)"]).
    vmins : list of float, optional
        Minimum values for colormaps (default: [-0.8]).
    vmaxes : list of float, optional
        Maximum values for colormaps (default: [0.8]).
    subplot_kw : dict, optional
        Keywords for creating subplots (default: {'projection': ccrs.PlateCarree()}).
    ssha_plot_kw : dict, optional
        Scatter plot options for SSH anomalies (default: marker and style settings).
    cmaps : list of colormap, optional
        List of colormaps for each variable (default: [cm.balance]).
    dpi : int, optional
        Resolution of the plot (default: 100 dpi).
    set_extent : bool, optional
        Whether to limit the map's extent (default: False).
    extent_lims : list of float, optional
        Bounding box for the map: [min_lon, max_lon, min_lat, max_lat] (default: global extent).
    axes : list, optional
        Existing axes to plot on; if None, a new figure is created (default: [None]).
    plot_bathymetry : bool, optional
        Whether to include bathymetry in the plot (default: True).

    Returns
    -------
    axs : list of matplotlib.axes
        Axes containing the plots.
    """
    
    # Load data (14.8 MB file)
    depths_str, shp_dict = load_bathymetry(
        'https://naturalearth.s3.amazonaws.com/' +
        '10m_physical/ne_10m_bathymetry_all.zip')

    # Construct a discrete colormap with colors corresponding to each depth
    depths = depths_str.astype(int)
    N = len(depths)
    nudge = 0.01  # shift bin edge slightly to include data
    boundaries = [min(depths)] + sorted(depths+nudge)  # low to high
    norm = matplotlib.colors.BoundaryNorm(boundaries, N)
    blues_cm = matplotlib.colormaps['Blues_r'].resampled(N)
    colors_depths = blues_cm(norm(depths))
    
    # Add labels to keep track of specific swaths
    swath_labels = []
    for swath in swaths:
        #label_lat = np.around(swath.latitude[-1,:].min().values,2)
        #label_lon = np.around(swath.longitude[-1,:].min().values,2)
        label_lat = np.around(swath.latitude[:,:].mean().values,2)
        label_lon = np.around(swath.longitude[:,:].mean().values,2)        
        if label_lon > 180:
            label_lon = label_lon - 360
        label_cycle = swath.cycle
        label_pass = swath.pass_ID
        if "time" in swath:
            label_time = swath.time.mean().values.astype(str)[:19]
        else:
            label_time = "" 
        swath_labels.append({"label_lat":label_lat,
                             "label_lon":label_lon,
                             "label_cycle":label_cycle,
                             "label_pass":label_pass,
                             "label_time":label_time
                            })

    # Initialize new figure if you are running things from scratch
    if axes[0] == None:
        fig, axs = plt.subplots(1,len(fields),figsize=(10*len(fields),20),subplot_kw=subplot_kw,dpi=dpi)
        # Using the get_axes() method returns the figure axes as a list...
        axs = fig.get_axes()

    # Add your new plots to a subfigure object if you were given one
    else:
        axs = axes
    print(axs)
    for i, ax in enumerate(axs):
        # Add some geographic features
        #ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        
        # Add bathymetry
        # Iterate and plot feature for each depth level
        if plot_bathymetry:
            for j, depth_str in enumerate(depths_str):
                ax.add_geometries(shp_dict[depth_str].geometries(),
                                  crs=ccrs.PlateCarree(),
                                  color=colors_depths[j])
        ax.add_feature(cfeature.LAND, edgecolor='none', facecolor='lightgray',zorder=0)
    
        # Normalize the colorbar to keep it constant between plots
        mynorm = plt.Normalize(vmin=vmins[i], vmax=vmaxes[i], clip=True)
        sm = plt.cm.ScalarMappable(cmap=cm.balance, norm=mynorm)
        
        # Plot initial swath for colorbar
        cax = ax.scatter(swaths[0].longitude[:,:], swaths[0].latitude[:,:], c=swaths[0][fields[i]][:,:], 
                         cmap=cmaps[i], **ssha_plot_kw, zorder=10, norm=mynorm)
        # add colorbar and gridlines
        cbar = plt.colorbar(cax, ax=ax, shrink=0.2, extend="both", pad=0.1)
        # Specify cbar title size / orientation
        cbar.set_label(cbar_titles[i],rotation=270,fontsize=30,labelpad=50)
        
        # Overlay remaining swaths
        for j, plt_swath in enumerate(swaths[1::]):
            print(j,end=",")
            ax.scatter(plt_swath.longitude[:,:], plt_swath.latitude[:,:], c=plt_swath[fields[i]][:,:], 
                       cmap=cmaps[i], **ssha_plot_kw, zorder=10+i, norm=mynorm)
    
        # If you're plotting quality flags, add a key
        if fields[i] == "quality_flag":
            ticks=np.arange(0,10,1)
            cbar.ax.set_yticks(ticks)
            cbar.ax.set_yticklabels(swaths[0].quality_flag.flag_meanings.split(" "))
    
        # Add labels
        if add_labels:
            for k, swath_label in enumerate(swath_labels):
                txt = ax.text(swath_label["label_lon"]-1,swath_label["label_lat"]+k%3-k%2,
                              (f"Cycle {swath_label["label_cycle"]} \n"
                              + f"Pass #{swath_label["label_pass"]}"
                              #+ f"{swath_label["label_time"][:10]} \n"
                              #+ f"{swath_label["label_time"][10:19]} " 
                                ),
                             fontsize=5,weight='bold',zorder=len(swaths)+30,color="w")
                txt.set_bbox(dict(facecolor='none', alpha=1, edgecolor='none'))

        # Title the subplot with the field you are plotting
        ax.set_title(f"{fields[i]}",fontsize=45)
 
        # Add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=1, color='gray', alpha=0.5, linestyle='--')
            
        # Set the extent of the map (x0, x1, y0, y1)
        if set_extent:
            ax.set_extent([extent_lims[0], extent_lims[1], extent_lims[2], extent_lims[3]], 
                           crs=ccrs.PlateCarree())

        plt.sca(ax)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # Add labels to the x- and y- axes    
        plt.ylabel("Latitude (deg)",fontsize=30)
        plt.xlabel("Longitude (deg)",fontsize=30)
        


        ax.set_title(f"Cycle {swath_labels[0]["label_cycle"]}" + f"\n" # + f"Pass #{swath_label["label_pass"]} \n"
                      + f"{swath_labels[0]["label_time"][:10]} {swath_labels[-1]["label_time"][10:19]} \n"
                       + f"{fields[i]}",            
                      fontsize=20)
    return axs
