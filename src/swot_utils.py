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
    xarray
    xrft

"""

import numpy as np
import scipy.signal as signal
import xarray as xr
import numpy as np
import requests
import scipy.interpolate

import xrft



def subset(data_in, lat_bounds, lon_bounds=None):
    """
    Subsets SWOT swaths based on latitude and optional longitude bounds.
    Note that this script assumes that the swath is oriented roughly 
    North-South since it uses simple array indexing and treats it like
    an NxM dimensiontal array. For more precise lat-lon-based subsetting 
    use the "xr_subset()" function below.
    
    Parameters
    ----------
    data_in : xarray.Dataset
        Swath xarray dataset, must contain 'latitude' and 'longitude' coordinates.
    lat_bounds : iterable
        Two-element list or tuple specifying the north-south latitude range for subsetting.
    lon_bounds : iterable, optional
        Two-element list or tuple specifying the east-west longitude range for subsetting.
        If not provided, subsetting is only done by latitude.
    
    Returns
    -------
    subset_data : xarray.Dataset or None
        Subset of the input dataset, or None if no valid subset can be created.
    """
    # Calculate cross-swath averaged latitudes and handle NaN values
    lat = np.nanmean(data_in['latitude'].load().values, axis=-1)
    lat = np.where(np.isnan(lat), 100, lat)  # Replace NaN values with 100 to avoid errors

    # Extract latitude bounds
    l0, l1 = lat_bounds

    # Find indices corresponding to the closest mean latitude values within bounds
    j0 = np.where(np.abs(lat - l0) == np.abs(lat - l0).min())[0][0]
    j1 = np.where(np.abs(lat - l1) == np.abs(lat - l1).min())[0][0]

    # Ensure the indices are in increasing order (flip if necessary)
    if j0 > j1:
        j0, j1 = j1, j0

    # If bounding indices are the same, return None (no valid subset)
    if j0 == j1:
        print(f"No data found in lat bounds")
        return None

    # Perform similar processing for longitude bounds, if provided
    if lon_bounds is not None:
        lon = np.nanmean(data_in['longitude'].load().values, axis=0)
        lon = np.where(np.isnan(lon), 100, lon)  # Replace NaN values
        l0, l1 = lon_bounds
        i0 = np.where(np.abs(lon - l0) == np.abs(lon - l0).min())[0][0]
        i1 = np.where(np.abs(lon - l1) == np.abs(lon - l1).min())[0][0]
        if i0 > i1:
            i0, i1 = i1, i0
        if i0 == i1:
            print(f"No data found in lon bounds")
            return None
    else:
        # If no longitude bounds provided, set indices to None
        i0 = None
        i1 = None

    # Subset variables that share the latitude (and optionally longitude) dimensions
    subset_vars = {}
    for varname, var in data_in.data_vars.items():
        if var.dims[0] not in data_in.latitude.dims:
            # Adding this step to remove any random variables that don't share
            # coordinates with the SSHA data. This includes some variables in
            # the Expert data (2/19/2025) like "i_num_pixels" and "i_num_line" 
            # whose dimensions are indexed differently (i.e. wrt to the distance to 
            # the nadir altimeter) rather than to the swath crosstrack / alongtrack 
            # coordinates.
            pass
        if (len(var.dims) == 2) and (lon_bounds is not None):  # Two-dimensional variables (lat-lon dependent)
            subset_vars[varname] = var[j0:j1, i0:i1]
        else:
            # For other variables, subset only by latitude
            subset_vars[varname] = var[j0:j1]
            
    # Combine the subset variables into a new xarray Dataset
    subset_data = xr.Dataset(subset_vars, attrs=data_in.attrs)
    
    return subset_data



def xr_subset(data_in, lat_bounds, lon_bounds=None):
    """
    Subsets an xarray dataset based on latitude and longitude bounds.
    This script is for use on any arbitrary xarray "swath", i.e. 
    a field with latitude and longitude coordinates. It uses xarray.where()
    and requires loading everyting in xarray format.

    Parameters
    ----------
    data_in : xarray.Dataset
        Swath xarray dataset, must contain 'latitude' and 'longitude' coordinates.
    lat_bounds : iterable
        Two-element list or tuple specifying the north-south latitude range for subsetting.
    lon_bounds : iterable
        Two-element list or tuple specifying the east-west longitude range for subsetting.
    
    Returns
    -------
    subset_data : xarray.Dataset or None
        Subset of the input dataset, or None if no valid subset can be created.
    """
    # Create a boolean mask for latitude, selecting values within the given latitude bounds.
    mask_lat = (data_in.latitude >= min(lat_bounds)-1) & (data_in.latitude <= max(lat_bounds)+1)

    if lon_bounds != None:
        # Create a boolean mask for longitude, selecting values within the given longitude bounds.
        mask_lon = (data_in.longitude >= min(lon_bounds)-1) & (data_in.longitude <= max(lon_bounds)+1)
    else:
        mask_lon = None

    # Initialize the cropped dataset as None to handle cases where no data is found.
    cropped_data = None
    
    try:
        # Apply the latitude and longitude masks to the dataset using xarray's where() function.
        # The drop=True argument removes values that do not meet the mask criteria.
        cropped_data = data_in.where(mask_lat.compute(), drop=True)
        if mask_lon != None:
            # Do longitudinal cropping if you need to
            cropped_data = data_in.where(mask_lon.compute(), drop=True)
    except Exception as e:  # Catch any errors that occur during the subsetting process.
        # Print error messages if no data is found within the specified bounds.
        print(f"Unable to find data in latlon bounds lat: {lat_bounds}, lon: {lon_bounds}")
        print(f"Exception: {e}")
                
    # Return the subset dataset, or None if an error occurred or no data matched the filters.
    return cropped_data



def compute_power_spectra_xrft(cycle_data, subset=False, lim0=1, lim1=519, assert_parseval=False, field="ssha_unedited"):
    """
    Compute power spectra for SWOT SSH data using xarray and xrft.

    Parameters
    ----------
    cycle_data : list of xarray.Dataset
        List of swath datasets containing the field to analyze.
    subset : bool, optional
        If True, apply subsetting to the swath data in the cross-track direction. Default is False.
    lim0 : int, optional
        Number of indices to ignore from the beginning of the cross-swath direction. Default is 1.
    lim1 : int, optional
        Number of indices to ignore from the end of the cross-swath direction. Default is 519.
    assert_parseval : bool, optional
        If True, print Parseval's theorem checks for energy conservation during FFT. Default is False.
    field : str, optional
        The field in the dataset to compute power spectra for. Default is "ssha_unedited".

    Returns
    -------
    tmp_freqs : list of ndarray
        List of frequency arrays for each swath.
    tmp_amps : list of ndarray
        List of power spectral density arrays for each swath.
    """
    tmp_freqs, tmp_amps = [], []  # Initialize lists to store results

    for swath_ds in cycle_data:
        # Extract the field data and convert to centimeters
        swath = swath_ds[field].values[:, :] * 100

        # Subset the swath data in the cross-swath direction if required
        if subset:
            swath[:, :lim0] = np.nan  # Mask the first `lim0` columns
            swath[:, lim1:520 - lim1] = np.nan  # Mask the middle columns
            swath[:, 520 - lim0:] = np.nan  # Mask the last `lim0` columns

        # Mask out NaN columns and retain valid columns
        msk = np.isnan(swath.mean(axis=0))
        swath = swath[:, ~msk]

        # Transpose swath for FFT (now rows represent cross-swath data)
        swath = swath.T

        # Initialize arrays to store frequency and power spectral density for this swath
        freqs = np.zeros(swath.shape)
        psds = np.zeros(swath.shape)

        # Compute the FFT and power spectrum for each cross-swath row
        for i in range(len(swath)):
            swath_i = swath[i, :]  # Extract one row
            Nx = swath_i.size  # Number of points
            dx = 0.25  # Spacing between points
            da = xr.DataArray(swath_i, dims="x", coords={"x": dx * np.arange(0, swath_i.size)})

            # Compute the Discrete Fourier Transform
            FT = xrft.dft(da, dim="x", true_phase=True, true_amplitude=True)

            # Compute the power spectrum
            ps = xrft.power_spectrum(da, dim="x")

            # Store frequencies and power spectral densities
            freqs[i, :] = FT["freq_x"].values
            psds[i, :] = ps.values

            if assert_parseval:
                # Perform Parseval's theorem checks
                print("Parseval's theorem directly from FFT:")
                print((np.abs(da) ** 2).sum() * dx)
                print((np.abs(FT) ** 2).sum() * FT["freq_x"].spacing)
                print()
                print("Parseval's theorem from power spectrum:")
                print(ps.sum())
                print((np.abs(da) ** 2).sum() * dx)
                print()

        # Append results for the current swath
        tmp_freqs.append(freqs)
        tmp_amps.append(psds)

    # Return the computed spectra
    return tmp_freqs, tmp_amps
