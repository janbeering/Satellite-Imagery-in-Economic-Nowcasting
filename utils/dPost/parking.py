# %%
# imports
import geopandas as gpd
import datetime

def parking_vehicles(roads_path, path,  AoI, crs_Kyiv, distance_to_road):
    """"""
    observations_gdf = gpd.read_csv(f'{path}/dist/result_csv/res_2024-12-12_10-14-02.shp')  # Replace with your actual file path
    roads_gdf = gpd.read_file(f'{roads_path}/AoI_Roads.shp')

    #filter obersvations by AoI
    observations_gdf = observations_gdf[observations_gdf.intersects(AoI)]

    #filter observations by labels. 0 and 1 are cars and trucks
    observations_gdf = observations_gdf[observations_gdf['labels'].isin([0, 1])]

    # Apply the function to each observation
    observations_gdf['nearest_road'] = observations_gdf.geometry.to_crs(crs_Kyiv).apply(find_nearest_road, roads_gdf.to_crs(crs_Kyiv))
    observations_gdf['parking'] =  observations_gdf['nearest_road'] < distance_to_road

    return observations_gdf

def find_nearest_road(point, roads_gdf):
    """"""
    # Calculate distances from the point to all roads
    distances = roads_gdf.distance(point) #https://epsg.io/?q=Ukraine%20kind:PROJCRS&page=2
    
    # Get the index of the nearest road
    nearest_idx = distances.idxmin()
    #print(nearest_idx, distances[nearest_idx])
    
    return distances[nearest_idx]

def create_grouped_results(observations_gdf):
    """"""
    combination_counts = observations_gdf.groupby(['labels', 'image_code', 'parking']).size().reset_index(name='counts')
    print(combination_counts)
    combination_counts.to_excel(f'/insert/own/path/grouped_results_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx')
