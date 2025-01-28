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
    pyresample
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
    Subset data using Jinbo's script (pulled out of L2 class in the SWOT_Open_Toolkit)
    
    Parameters
    ----------
    data_in : xarray dataset
        Swath xarray dataset, must contain lat and lon coordinates
    lat_bounds : iterable, 
        North - South boundaries at which to subset
    lon_bounds : iterable,
        East - West boundaries at which to subset        
    
    Returns
    -------
    
    """
    # Pull cross-swath averaged latitudes and find bounding indices
    lat=np.nanmean(data_in['latitude'].load().values,axis=-1)
    # Set Nan values to 100
    lat=np.where(np.isnan(lat),100,lat)
    l0,l1=lat_bounds
    # Find the index of the closest mean 
    # cross-track latitude to the lat bounds
    j0=np.where(np.abs(lat-l0)==np.abs(lat-l0).min())[0][0]
    j1=np.where(np.abs(lat-l1)==np.abs(lat-l1).min())[0][0]
    # Flip order if swath is descending
    if j0>j1:
        j0,j1=j1,j0
    # Return empty array if bounding indices are the same
    if j0==j1:
        return None

    # If you also added longitudinal boundaries, repeat the earlier steps
    # in the zonal direction
    if lon_bounds is not None:
        lon=np.nanmean(data_in['longitude'].load().values,axis=0)
        lon=np.where(np.isnan(lon),100,lon)
        l0,l1=lon_bounds
        i0=np.where(np.abs(lon-l0)==np.abs(lon-l0).min())[0][0]
        i1=np.where(np.abs(lon-l1)==np.abs(lon-l1).min())[0][0]
        if i0>i1:
            i0,i1=i1,i0
        if i0==i1:
            return None
    else:
        i0=None
        i1=None
            
    # Subset all variables that share the latitude dimension
    subset_vars = {}
    for varname, var in data_in.data_vars.items():
        if len(var.dims)==2:
            subset_vars[varname] = var[j0:j1,i0:i1]
        else:
            # Else just do lat bounds, perhaps should change this..
            subset_vars[varname] = var[j0:j1]
    
    # Combine the subset variables into a new dataset
    subset_data = xr.Dataset(subset_vars, attrs=data_in.attrs)
    
    return subset_data




def compute_power_spectra_xrft(cycle_data,subset=False,lim0=1,lim1=519,assert_parseval=False,field="ssha_unedited"):
    """
    Inputs
    ------
    cycle_data: Iterable containing swaths for extraction. I'm assuming that the swaths are 
                ndarray-like with shape swath.shape = [along-track direction, cross-track direction] 
    clean_swath_nan: Bollean, if "True" filter out swath data in the cross-track direction using 
                     "lim0" and "lim1"
    lim0: Integer, cross-swath index distance from the outer edge of each swath 
                    to drop when calculating the spectra
    lim1: Integer, cross-swath index distance from the outer edge of each swath 
                    to drop when calculating the spectra

    Outputs
    ------
    tmp_freqs: 
    tmp_amps:
    
    """


    tmp_freqs, tmp_amps = [], []
    
    for swath_ds in cycle_data:
        # Convert swath to cm
        swath = swath_ds[field].values[:,:]*100
        
        # Nan out columns you don't want
        if subset:
            swath[:,:lim0] = np.nan
            swath[:,lim1:520-lim1] = np.nan
            swath[:,520-lim0:] = np.nan
            
        # Mask out NaNed columns (sides/middle os swath) 
        msk = np.isnan(swath.mean(axis=0))
        swath = swath[:,~msk]
    
        # I'm assuming the dataset shape here...
        swath = swath.T
    
        # Create output arrays
        freqs = np.zeros(swath.shape)
        psds = np.zeros(swath.shape)
    
        # Perform along-swath FFT
        for i in range(len(swath)):
            swath_i = swath[i,:]
            Nx = swath_i.size
            dx = .25
            da = xr.DataArray(swath_i,dims="x",coords={"x": dx * np.arange(0,swath_i.size)},)
    
            # Calculate FFT
            FT = xrft.dft(da, dim="x", true_phase=True, true_amplitude=True)
            # Calculate power spectrum
            ps = xrft.power_spectrum(da, dim="x")
    
            # Record
            freqs[i,:] = FT["freq_x"].values
            psds[i,:] = ps.values
            
            if assert_parseval:
                ###############
                # Assert Parseval's using xrft.dft
                ###############
                print("Parseval's theorem directly from FFT:")
                print((np.abs(da) ** 2).sum() * dx)
                print((np.abs(FT) ** 2).sum() * FT["freq_x"].spacing)
                print()
                                       
                ###############
                # Assert Parseval's using xrft.power_spectrum with scaling='density'
                ###############
                print("Parseval's theorem from power spectrum:")        
                print(ps.sum())
                print((np.abs(da) ** 2).sum() * dx )
                print()
        
        # Record data for each swath, don't average in the cross-swath direction yet
        tmp_freqs.append(freqs)
        tmp_amps.append(psds)
    
    # Return the spectra
    return tmp_freqs, tmp_amps

