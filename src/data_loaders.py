#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains routines for opening locally-saved SWOT (Surface Water and Ocean Topography) data.

Functions:
1. remap_quality_flags: Reassigns discrete quality flags in the data to a smaller, simpler range.
2. load_cycle: Loads and processes SWOT data for a specified cycle and optionally filters it.

Author: Tatsu, comments partially by ChatGPT :,)
Date: First version: 1.23.2025

Dependencies:
    - xarray
    - numpy
"""

import os
import xarray as xr
import swot_utils  # Custom utilities for SWOT data

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Function: remap_quality_flags
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def remap_quality_flags(swath):
    """
    Remaps quality flags in the SWOT dataset to simplified numeric categories.

    Parameters
    ----------
    swath : xarray.Dataset
        The SWOT dataset containing the 'quality_flag' variable.

    Returns
    -------
    xarray.Dataset
        The modified dataset with remapped quality_flag values.

    Notes
    -----
    - The function maps the following original flag values to a new range:
        5   -> 1
        10  -> 2
        20  -> 3
        30  -> 4
        50  -> 5
        70  -> 6
        100 -> 7
        101 -> 8
        102 -> 9
    - This simplifies plotting and interpretation of quality flags.
    """
    # Check if the 'quality_flag' variable exists in the dataset
    if not "quality_flag" in swath:
        return

    # Access the 'quality_flag' variable
    flags = swath.quality_flag

    # Remap the quality flag values using direct value replacement
    flags.values[flags.values == 5.] = 1
    flags.values[flags.values == 10.] = 2
    flags.values[flags.values == 20.] = 3
    flags.values[flags.values == 30.] = 4
    flags.values[flags.values == 50.] = 5
    flags.values[flags.values == 70.] = 6
    flags.values[flags.values == 100.] = 7
    flags.values[flags.values == 101.] = 8
    flags.values[flags.values == 102.] = 9

    # Update the 'quality_flag' variable in the dataset
    swath.quality_flag.values = flags.values
    
    return swath


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Function: load_cycle
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_cycle(path, cycle="002", pass_ids=None, fields=None, subset=False, lat_lims=[-90, 90]):
    """
    Loads SWOT data for a specific cycle from locally stored NetCDF files.

    Parameters
    ----------
    path : str
        Path to the root directory containing SWOT data cycles.
    cycle : str, optional
        The cycle number to load (default is "002").
    pass_ids : list of str, optional
        Specific pass IDs to load. If None, all passes are loaded (default: None).
    fields : list of str, optional
        Fields (variables) to extract from the dataset. If None, all fields are loaded (default: None).
    subset : bool, optional
        Whether to subset the data spatially by latitude (default: False).
    lat_lims : list of float, optional
        Latitude range [min_lat, max_lat] to use for subsetting if `subset` is True (default: [-90, 90]).

    Returns
    -------
    list of xarray.Dataset
        A list of datasets, one for each loaded SWOT pass.

    Notes
    -----
    - Datasets are filtered by cycle and optionally by pass ID.
    - Remaps 'quality_flag' values for easier discrete plotting.
    """
    # Check if the specified cycle directory exists
    if not os.path.exists(f"{path}/cycle_{cycle}"):
        print(f"Can't find path {path}/cycle_{cycle}")
        return []  # Return an empty list if the directory does not exist
    
    # If no pass IDs are provided, load all passes in the cycle
    if pass_ids is None:
        # List all NetCDF files in the cycle directory
        swot_passes = [f for f in os.listdir(f"{path}/cycle_{cycle}") if ".nc" in f]
    else:
        # If pass IDs are specified, filter for matching files
        swot_passes = []
        for pass_id in pass_ids:
            passes = [f for f in os.listdir(f"{path}/cycle_{cycle}") if f"Unsmoothed_{cycle}_{pass_id}" in f]
            swot_passes += passes  # Append matching files to the list
    
    # Sort the passes by pass ID (6th element in the file name, after splitting by '_')
    swot_passes = sorted(swot_passes, key=lambda x: int(x.split("_")[6]))

    # Initialize an empty list to store the loaded passes
    passes = []

    # Process each pass file
    for swot_pass in swot_passes:
        print(f"Loading {swot_pass}")
        try:
            # Open the NetCDF file as an xarray Dataset
            swath = xr.open_dataset(f"{path}/cycle_{cycle}/{swot_pass}")

            # If specific fields are requested, extract only those
            if fields is None:
                fields = list(swath.variables)  # Load all variables if none are specified
            swath = swath[fields]

            # Subset the data by latitude if requested
            if subset:
                swath = swot_utils.subset(swath, lat_lims)

            # Add cycle and pass ID as metadata attributes
            swath = swath.assign_attrs(
                cycle=f"{cycle}",
                pass_ID=f"{swot_pass.split('_')[6]}"
            )

            # Remap the quality flags for discrete plotting
            if "quality_flag" in fields:
                swath = remap_quality_flags(swath)

            # Append the processed swath to the list
            passes.append(swath)
        except Exception as e:
            print("Whoops, can't open dataset")
            print("An error occurred:", e)

    # Return the list of loaded and processed passes
    return passes
