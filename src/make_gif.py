import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

# Plot and save as GIF
def create_gif_from_xarray(dataarray, gif_name="output.gif", fps=5, vmin=None, vmax=None, cmap="Spectral_r"):
    """
    Create a .gif from a time sequence of 2D xarray DataArray.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The 2D+time data to be animated (must have dimensions "time", "longitude", "latitude").
    gif_name : str, optional
        Name of the output .gif file (default is "output.gif").
    fps : int, optional
        Frames per second for the GIF (default is 5).
    vmin, vmax : float, optional
        Minimum and maximum values for consistent colormap scaling across frames.
    """
    # Ensure the data has the required dimensions
    if not all(dim in dataarray.dims for dim in ["time", "longitude", "latitude"]):
        raise ValueError("The DataArray must have dimensions ('time', 'longitude', 'latitude').")

    # Get the colormap limits if not provided
    if vmin is None:
        vmin = dataarray.min().item()
    if vmax is None:
        vmax = dataarray.max().item()

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot the first timestep to get the colorbar
    im = ax.scatter(
        dataarray.longitude.values,
        dataarray.latitude.values,
        c=dataarray.isel(time=0).values,
        s=1,
        cmap=cmap,
        levels=np.linspace(vmin,vmax,100),
    )
    fig.colorbar(im, ax=ax, orientation="vertical", label=dataarray.name)
    
    # Define the update function for each frame
    def update(frame):
        ax.clear()  # Clear the previous frame
        ax.set_title(f"Time: {dataarray.time.values[frame]}")
        ax.scatter(
            dataarray.longitude.values,
            dataarray.latitude.values,
            c=dataarray.isel(time=frame-1).values,
            s=1,
            levels=np.linspace(vmin,vmax,100),
            alpha=0.6
        )
        im = ax.scatter(
            dataarray.longitude.values,
            dataarray.latitude.values,
            c=dataarray.isel(time=frame).values,
            s=1,
            levels=np.linspace(vmin,vmax,100),
        )
        return im

    # Create an animation writer
    writer = PillowWriter(fps=fps)
    writer.setup(fig, gif_name)

    # Generate the GIF frame by frame, starting from the second timestep
    for frame in range(1, len(dataarray.time)):
        update(frame)
        writer.grab_frame()

    # Finalize the writer and save the GIF
    writer.finish()

    print(f"GIF saved as '{gif_name}'")
