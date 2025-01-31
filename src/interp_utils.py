#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains routines for interpolating and regridding SWOT (Surface Water and Ocean Topography) and VIIRS 
(Visible Infrared Imaging Radiometer Suite, onboard the NOAA-20, -21, and -NPP satellites) swath data. 
It includes functions for coordinate transformations between geodetic latitude-longitude-altitude and 
local ENU (East-North-Up) coordinates. The code is adapted from the interp_utils package in Scott Martin's NeurOST 
project, with references to PyProj transformations and a Stack Overflow discussion on ENU conversions.

This code is pretty flexible and can be used for regridding any lat-lon spatial dataset, for example llc4320.

Author: Tatsu
Date: First version: 1.27.2025

Dependencies:
    numpy
    scipy
    pyproj
    xarray
"""

# Import necessary libraries
import numpy as np
import pyproj  # For geodetic coordinate transformations
import xarray as xr  # For handling multidimensional datasets
import scipy.spatial.transform  # For 3D rotation matrices
from scipy import stats # For binning the swath on a 2D domain
from datetime import date, timedelta  # For handling dates
import os  # For interacting with the operating system


# Define pyproj transformer objects for coordinate transformations
# These objects handle transformations between geodetic coordinates (lat, lon, alt)
# and Earth-Centered-Earth-Fixed (ECEF) coordinates.
TRANSFORMER_LL2XYZ = pyproj.Transformer.from_crs(
    {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},  # Source CRS: Geodetic
    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}   # Target CRS: ECEF
)

TRANSFORMER_XYZ2LL = pyproj.Transformer.from_crs(
    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},  # Source CRS: ECEF
    {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'}   # Target CRS: Geodetic
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ENU to lat/lon transformation
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def xyz2ll(x, y, z, lat_org, lon_org, alt_org, transformer1=TRANSFORMER_LL2XYZ, transformer2=TRANSFORMER_XYZ2LL):
    """
    Converts local ENU (East-North-Up) coordinates back to geodetic latitude and longitude.

    Parameters
    ----------
    x, y, z : ndarray
        Arrays representing ENU coordinates relative to the origin.
    lat_org, lon_org, alt_org : float
        Geodetic coordinates (latitude, longitude, altitude) of the ENU origin.
    transformer1 : pyproj.Transformer, optional
        Transformer for geodetic to ECEF conversion (default: TRANSFORMER_LL2XYZ).
    transformer2 : pyproj.Transformer, optional
        Transformer for ECEF to geodetic conversion (default: TRANSFORMER_XYZ2LL).

    Returns
    -------
    lat, lon : ndarray
        Arrays of geodetic latitude and longitude corresponding to the input ENU coordinates.
    """

    # Convert the ENU origin's geodetic coordinates to ECEF coordinates (https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system)
    # The origin is the reference point for the local tangent plane.
    x_org, y_org, z_org = transformer1.transform(lon_org, lat_org, alt_org, radians=False)
    ecef_org = np.array([[x_org, y_org, z_org]]).T

    # Create a rotation matrix to transform between ECEF and ENU coordinates (https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)
    # This involves two rotations:
    # 1. Rotating about the X-axis to account for the latitude.
    rot1 = scipy.spatial.transform.Rotation.from_euler('x', -(90 - lat_org), degrees=True).as_matrix()
    # 2. Rotating about the Z-axis to account for the longitude.
    rot3 = scipy.spatial.transform.Rotation.from_euler('z', -(90 + lon_org), degrees=True).as_matrix()
    # Combine the rotations into a single transformation matrix.
    rotMatrix = rot1.dot(rot3)

    # Transform ENU coordinates to ECEF by applying the inverse rotation matrix.
    ecefDelta = rotMatrix.T.dot(np.stack([x, y, np.zeros_like(x)], axis=-1).T)
    # Add the ECEF origin offset to obtain the absolute ECEF coordinates.
    ecef = ecefDelta + ecef_org

    # Convert ECEF coordinates back to geodetic latitude, longitude, and altitude.
    lon, lat, alt = transformer2.transform(ecef[0, :], ecef[1, :], ecef[2, :], radians=False)

    # only return lat, lon since we're interested in points on Earth. 
    # N.B. this amounts to doing an inverse stereographic projection from ENU to lat, lon so shouldn't be used to directly back calculate lat, lon from tangent plane coords
    # this is instead achieved by binning the data's lat/long variables onto the grid in the same way as is done for the variable of interest
    return lat, lon

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Lat/lon to ENU transformation
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def ll2xyz(lat, lon, alt, lat_org, lon_org, alt_org, transformer=TRANSFORMER_LL2XYZ):
    """
    Converts geodetic latitude, longitude, and altitude to local ENU (East-North-Up) coordinates.

    Parameters
    ----------
    lat, lon, alt : ndarray
        Arrays representing geodetic coordinates of the points to be transformed.
    lat_org, lon_org, alt_org : float
        Geodetic coordinates of the ENU origin.
    transformer : pyproj.Transformer, optional
        Transformer for geodetic to ECEF conversion (default: TRANSFORMER_LL2XYZ).

    Returns
    -------
    X, Y, Z : ndarray
        Arrays representing the ENU coordinates relative to the origin.
    """

    # Convert geodetic coordinates to ECEF  (https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system)
    x, y, z = transformer.transform(lon, lat, np.zeros_like(lon), radians=False)
    # Convert the ENU origin's geodetic coordinates to ECEF.
    x_org, y_org, z_org = transformer.transform(lon_org, lat_org, alt_org, radians=False)

    # Compute the vector from the origin to the points in ECEF coordinates.
    vec = np.array([[x - x_org, y - y_org, z - z_org]]).T

    # Create a rotation matrix to transform from ECEF to ENU (https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)
    rot1 = scipy.spatial.transform.Rotation.from_euler('x', -(90 - lat_org), degrees=True).as_matrix()
    rot3 = scipy.spatial.transform.Rotation.from_euler('z', -(90 + lon_org), degrees=True).as_matrix()
    rotMatrix = rot1.dot(rot3)

    # Apply the rotation matrix to convert ECEF vectors to ENU coordinates.
    enu = rotMatrix.dot(vec).T

    # Extract the X (East), Y (North), and Z (Up) components of the ENU coordinates.
    X, Y, Z = enu[0, :, 0], enu[0, :, 1], enu[0, :, 2]

    return X, Y, Z
    

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# General gridding function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def grid_field_enu(x, y, ssh, n, L_x, L_y):
    """
    Interpolates a field onto a 2D grid in the ENU tangent plane using a statistical binning approach.
    Source: NeurOST (Scott Martin, UW)

    Parameters
    ----------
    x, y : ndarray
        Arrays representing the ENU coordinates of the data points. These are the horizontal axes
        (East and North) in the tangent plane.
    ssh : ndarray
        The input field to be gridded (e.g., sea surface height anomalies).
    n : int
        The number of bins along each axis of the output grid.
    L_x, L_y : float
        The extents (in meters) of the gridded region along the x (East) and y (North) axes.

    Returns
    -------
    gridded_data : ndarray
        A 2D array representing the gridded field, where each grid cell contains the mean value of
        the input field (`ssh`) for all data points within that cell.
        If no data points fall into a given cell, the value in that cell will be NaN.
    """
    # Use scipy's binned_statistic_2d to calculate the mean of `ssh` within 2D grid bins.
    # - `statistic='mean'` computes the average value in each bin.
    # - `bins=n` specifies the number of bins along each axis.
    # - `range=[[-L_x/2, L_x/2], [-L_y/2, L_y/2]]` defines the spatial extent of the grid.

    gridded_data = np.rot90(
        stats.binned_statistic_2d(
            x, y, ssh,
            statistic='mean',
            bins=n,
            range=[[-L_x / 2, L_x / 2], [-L_y / 2, L_y / 2]]
        )[0], k=-1
    )

    # Data comes out flipped for some reason
    gridded_data = np.flip(gridded_data,axis=1)
    
    # Rotate the resulting 2D array by 90 degrees to align the output with expected orientation.
    # This is often needed because binned_statistic_2d may produce an output with an inverted
    # or transposed axis order.

    return gridded_data
    

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Wrapper for resampling lat/lon fields to a gridded ENU projection
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def grid_everything(swath_data, lat0, lon0, n=256, L_x=256e3, L_y=256e3):
    """
    Interpolates swath data onto a regular grid using an ENU projection.
    Thanks ChatGPT for writing the comments :,) 

    Parameters
    ----------
    swath_data : xarray.Dataset or xarray.DataArray
        The input data to be interpolated. If it is an xarray.Dataset, all variables
        will be gridded. If it is an xarray.DataArray, only the single variable will be gridded.
        Must include 'latitude' and 'longitude' coordinates.
    lat0 : float
        The latitude to use as the origin for the ENU projection.
    lon0 : float
        The longitude to use as the origin for the ENU projection.
    n : int, optional
        The number of bins in the output grid along each dimension (default: 256).
    L_x : float, optional
        The longitudinal range (extent) of the output grid in meters (default: 256,000 m).
    L_y : float, optional
        The latitudinal range (extent) of the output grid in meters (default: 256,000 m).

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        A gridded version of the input data:
        - If the input is a Dataset, returns a Dataset with all variables gridded.
        - If the input is a DataArray, returns a DataArray for the single variable.
        The output includes associated latitude, longitude, ENU X, and ENU Y coordinates.
    """
    # Extract latitude and longitude from the input dataset or data array
    lats = swath_data.latitude.values.flatten()
    lons = ((swath_data.longitude.values.flatten() % 180) - 180)

    # Project latitude and longitude coordinates onto an ENU coordinate system
    x, y, z = ll2xyz(lats, lons, 0, lat0, lon0, 0)

    # Generate regular cartesian points for grid coordinates
    x_c, y_c = np.linspace(-L_x / 2, L_x / 2, n), np.linspace(-L_y / 2, L_y / 2, n)
    
    # Generate gridded lat/lon coordinates for the ENU tangent plane
    xx_c, yy_c = np.meshgrid(x_c, y_c)
    lat_c, lon_c = xyz2ll(xx_c.flatten(), yy_c.flatten(), 0, lat0, lon0, 0)
    # reshape
    lat_c, lon_c = np.reshape(lat_c,yy_c.shape), np.reshape(lon_c,xx_c.shape)

    # Check if the input is a Dataset or a DataArray
    # If the input is a Dataset, grid each variable
    if isinstance(swath_data, xr.Dataset):
        # Initialize a dictionary to hold gridded variables
        gridded_vars = {}

        # Process each variable in the Dataset
        for var_name, data_array in swath_data.data_vars.items():
            # Grid the variable
            gridded_data = grid_field_enu(x, y, data_array.values.flatten(), n, L_x, L_y)
            gridded_vars[var_name] = (["x", "y"])

        # Grid latitude, longitude, and ENU coordinates
        lat_gridded = grid_field_enu(x, y, lats, n, L_x, L_y)
        lon_gridded = grid_field_enu(x, y, lons, n, L_x, L_y)

        # Return a gridded xarray.Dataset
        return xr.Dataset(
            data_vars=gridded_vars,
            coords=dict(
                latitude=(["x", "y"], lat_gridded),
                longitude=(["x", "y"], lon_gridded),
                x=(["x"], x_c),
                y=(["y"], y_c),
            ),
        )
    # Elif the input is a Dataarray, grid the field
    elif isinstance(swath_data, xr.DataArray):
        # Interpolate the single data variable
        gridded_data = grid_field_enu(x, y, swath_data.values.flatten(), n, L_x, L_y)

        # Grid latitude, longitude, and ENU coordinates
        lat_gridded = grid_field_enu(x, y, lats, n, L_x, L_y)
        lon_gridded = grid_field_enu(x, y, lons, n, L_x, L_y)

        # Return a gridded xarray.DataArray
        return xr.DataArray(
            data=gridded_data,
            dims=["x", "y"],
            coords=dict(
                latitude=(["x", "y"], lat_gridded),
                longitude=(["x", "y"], lon_gridded),
                x=(["x"], x_c),
                y=(["y"], y_c),
            ),
        )
    else:
        raise TypeError("Input must be an xarray.Dataset or xarray.DataArray")

    return

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extra helper functions from NeursOST
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def normalise_ssh(ssh, mean_ssh, std_ssh):    
    return (ssh-mean_ssh)/std_ssh


def rescale_x(x, L_x, n):
    return (x + 0.5*L_x)*(n - 1)/L_x


def rescale_y(y, L_y, n): 
    return (-y + 0.5*L_y)*(n - 1)/L_y




    
