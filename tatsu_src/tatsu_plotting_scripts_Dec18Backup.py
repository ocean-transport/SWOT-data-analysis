import xarray as xr
import os
import numpy as np
from glob import glob

import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Use shpreader for bathymetry file
import cartopy.io.shapereader as shpreader

import cmocean.cm as cm



# A collection of quick and dirty scripts for Tatsu to use to look
# at some initial results

def load_bathymetry(zip_file_url):
    """
    A helper script to read from a zip file from Natural Earth 
    containing bathymetry shapefiles.
    
    """
    # Download and extract shapefiles
    import io
    import zipfile
    import os

    import requests
    
    # Download bathymetry if you don't have it already..
    if not os.path.exists("ne_10m_bathymetry_all/"):
        import requests
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("ne_10m_bathymetry_all/")
    
    # Read shapefiles, sorted by depth
    shp_dict = {}
    files = glob('ne_10m_bathymetry_all/*.shp')
    assert len(files) > 0
    files.sort()
    depths = []
    for f in files:
        depth = '-' + f.split('_')[-1].split('.')[0]  # depth from file name
        depths.append(depth)
        bbox = (-180, -90, 180, 90)  # (x0, y0, x1, y1)
        nei = shpreader.Reader(f, bbox=bbox)
        shp_dict[depth] = nei
    depths = np.array(depths)[::-1]  # sort from surface to bottom
    return depths, shp_dict

##########################################################################################
if __name__ == "__main__":
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

##########################################################################################

def remap_quality_flags(swath):
    """
    A simple script to remap quality flags >:(
    """

    if not "quality_flag" in swath:
        return

    flags = swath.quality_flag
    flags.values[flags.values==5.] = 1
    flags.values[flags.values==10.] = 2
    flags.values[flags.values==20.] = 3
    flags.values[flags.values==30.] = 4
    flags.values[flags.values==50.] = 5
    flags.values[flags.values==70.] = 6
    flags.values[flags.values==100.] = 7
    flags.values[flags.values==101.] = 8
    flags.values[flags.values==102.] = 9

    swath.quality_flag.values = flags.values
    
    return swath



def plot_cycle(swaths,fields=["ssha"],title="Example Swaths August 2024 (Cycle 4)",cbar_titles=["SSHA (m)"],
               vmins=[-0.8],vmaxes=[0.8], subplot_kw={'projection': ccrs.PlateCarree()},
               ssha_plot_kw = {"transform":ccrs.PlateCarree(),
                              "s":1,"marker":".","alpha":1,"linewidths":0},
               cmaps=[cm.balance],
               save_fig=False,save_fig_name="swath.png",dpi=100,
               set_extent=False,extent_lims=[-180,180,-90,90],
               axis=None, plot_bathymetry=True,
              ):
    """
    A complicated script to make nice plots of swaths across a single cycle
    
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
        label_lat = np.around(swath.latitude[-1,:].min().values,2)
        label_lon = np.around(swath.longitude[-1,:].min().values,2)
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
    if axis == None:
        fig, axs = plt.subplots(1,len(fields),figsize=(10*len(fields),20),subplot_kw=subplot_kw,dpi=dpi)
        # Using the get_axes() method returns the figure axes as a list...
        axs = fig.get_axes()

    # Add your new plots to a subfigure object if you were given one
    else:
        axs = axis
        
    for i, ax in enumerate(axs):
        # Add some geographic features
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        ax.add_feature(cfeature.LAND, edgecolor='none', facecolor='lightgray')
        # Add bathymetry
        # Iterate and plot feature for each depth level
        if plot_bathymetry:
            for j, depth_str in enumerate(depths_str):
                ax.add_geometries(shp_dict[depth_str].geometries(),
                                  crs=ccrs.PlateCarree(),
                                  color=colors_depths[j])
    
        # Normalize the colorbar to keep it constant between plots
        mynorm = plt.Normalize(vmin=vmins[i], vmax=vmaxes[i], clip=True)
        sm = plt.cm.ScalarMappable(cmap=cm.balance, norm=mynorm)
        
        # Plot initial swath for colorbar
        cax = ax.scatter(swaths[0].longitude[:,:], swaths[0].latitude[:,:], c=swaths[0][fields[i]][:,:], 
                         cmap=cmaps[i], **ssha_plot_kw, zorder=10, norm=mynorm)
        # add colorbar and gridlines
        cbar = plt.colorbar(cax, ax=ax, shrink=0.4, extend="both", pad=0.1)
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
        for swath_label in swath_labels:
            txt = ax.text(swath_label["label_lon"],swath_label["label_lat"],
                          (f"Cycle {swath_label["label_cycle"]} \n"
                          + f"Pass #{swath_label["label_pass"]} \n"
                          + f"{swath_label["label_time"][:10]} \n"
                          + f"{swath_label["label_time"][10:19]} " ),
                         fontsize=10,weight='bold',zorder=len(swaths)+30)
            txt.set_bbox(dict(facecolor='white', alpha=1, edgecolor='k'))

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
        


    # Custom title depending on the number of swaths you are plotting
    if len(swaths) > 1:
        ax.set_title(f"Cycle {swath_labels[0]["label_cycle"]}" + f"\n" # + f"Pass #{swath_label["label_pass"]} \n"
                      + f"{swath_labels[0]["label_time"][:10]} {swath_labels[-1]["label_time"][10:19]} ",
                      fontsize=20)
    else:
        ax.set_title(f"Cycle {swath_labels[0]["label_cycle"]} Pass #{swath_labels[0]["label_pass"]} \n"
                       + f"{swath_labels[0]["label_time"][:10]} {swath_labels[0]["label_time"][10:19]} \n"
                       + f"{fields[i]}",
                       fontsize=20,pad=10)    

    #fig.tight_layout()

    #if save_fig:
    #    plt.savefig(f"movie_figs/{save_fig_name}.png")
    
    #plt.show()
    #plt.close()
    
    return fig
