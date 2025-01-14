import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_obs_roads(observations_gdf, roads_gdf, figure_path):
    """ Plots for all images with observations and roads"""
    for image in observations_gdf['image_code'].unique():
        print(image)

        fig, ax = plt.subplots(figsize=(20, 20))

        # Plot the roads (as lines) on the axis
        roads_gdf.plot(ax=ax, color='black', linewidth=0.5, label='Roads')

        # Plot the observations (as points) on the same axis
        observations_gdf[observations_gdf["image_code"] == image].plot(ax=ax, color='red', markersize=0.2, label='Observations')

        # Add a legend
        plt.title(f'Observations in Kyiv - Image {image}')
        plt.legend()
        plt.savefig(f'{figure_path}/roads_and_obs_{image}.png', dpi=300, bbox_inches='tight')
        # Show the plot
        plt.show()
        plt.clf()
        

def plot_Hist2D(observations_gdf, roads_gdf, figure_path, nbins):
    """ 
    In a first loop, the max and min values for all images are derived to ensure a consistent scale for all images.
    In the second loop, the 2D Histogramm plots of all images are made.
    """
    # First Loop
    vmin = 0
    vmax = 0
    for image in observations_gdf['image_code'].unique():
        fig, axes = plt.subplots(figsize=(50, 50))
        all = axes.hist2d(observations_gdf[observations_gdf['image_code'] == image].geometry.x , observations_gdf[observations_gdf['image_code'] == image].geometry.y , bins=nbins, cmap=plt.cm.BuGn_r)
        print(np.max(all[0]))
        if np.max(all[0]) > vmax:
            vmax = np.max(all[0])
        if np.min(all[0]) > vmin:
            vmin = np.min(all[0])
    
    # Second Loop
    for image in observations_gdf['image_code'].unique():
        fig, axes = plt.subplots(figsize=(50, 50))
        # 2D Histogram
        roads_gdf.plot(ax=axes, color='black', linewidth=1, label='Roads')

        h = axes.hist2d(observations_gdf[observations_gdf['image_code'] == image].geometry.x , observations_gdf[observations_gdf['image_code'] == image].geometry.y , bins=nbins, cmap=plt.cm.BuGn_r, vmin = vmin, vmax = vmax)
        
        cbar = fig.colorbar(h[3], ax=axes, location= 'bottom', shrink = 0.2, pad= 0.02)
        cbar.minorticks_on()
        plt.title(f'2D Hist in Kyiv - Image {image}')
        #plt.legend()
        plt.savefig(f'{figure_path}/2dHist_{image}.png', dpi=300, bbox_inches='tight')


def plot_KDE(observations_gdf, roads_gdf, figure_path, nbins):
    for image in observations_gdf['image_code'].unique():

        fig, axes = plt.subplots(figsize=(50, 50))
        obs_image = observations_gdf[observations_gdf['image_code'] == image]
        listarray = []
        for pp in obs_image.geometry:
            listarray.append([pp.x, pp.y])
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        k = gaussian_kde(np.array(listarray).T)
        xi, yi = np.mgrid[obs_image.geometry.x.min():obs_image.geometry.x.max():nbins*1j, obs_image.geometry.y.min():obs_image.geometry.y.max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
        roads_gdf.plot(ax=axes, color='black', linewidth=1, label='Roads')

        axes.set_title('Calculate Gaussian KDE')
        axes.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.BuGn_r)
        plt.title(f'KDE in Kyiv - Image {image}')
        plt.legend()
        plt.savefig(f'{figure_path}/KDE_{image}_grid.png', dpi=300, bbox_inches='tight')
       
        
def plot_intersection_area(roads_path, figures_path, intersection_wgs84):
    gdf_roads = gpd.read_file(f'{roads_path}/ukr_roads/gis_osm_roads_free_1.shp')
    gdf_roads = gdf_roads[gdf_roads["code"].isin([5111, 5112, 5113, 5131, 5132, 5133])] #include main roads for better orientation on the map

    kyiv_gdf = gpd.read_file(f'{roads_path}/ukr_roads/kyiv.geojson')
    kyiv_geometry = kyiv_gdf.union_all()  # Merge all geometries in the GeoJSON
    
    roads_gdf_filtered = gdf_roads[gdf_roads.intersects(kyiv_geometry)]
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Kyiv
    kyiv_gdf.plot(ax=ax, color='red', alpha=0.5, edgecolor='black', label='Kyiv')

    #Plot Roads
    roads_gdf_filtered.plot(ax=ax, color='black', linewidth=1, label='Roads')

    # Plot polygon
    x1, y1 = intersection_wgs84.exterior.xy
    ax.plot(x1, y1, marker='o', label='Intersection Area', color='blue')
    ax.fill(x1, y1, alpha=0.3, color='blue')

    # Add details to the plot
    ax.set_title("Intersection Area within Kyiv")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True)
    plt.savefig(f'{figures_path}/Intersection_Area.png', dpi=300, bbox_inches='tight')
    plt.show()