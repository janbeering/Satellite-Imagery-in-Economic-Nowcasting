# %%
# imports
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import transform
import pyproj
from functools import reduce

def intersection(coordinate_bounds, proj_wgs84, proj_utm):
    """ Derives and returns the intersection boundaries of all images, calculates the area. """
    project_toutm = pyproj.Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True).transform
    project_towgs84 = pyproj.Transformer.from_crs(proj_utm, proj_wgs84, always_xy=True).transform


    # Polygon with transformation from WGS84 to UTM to ensure correct area calculation
    polygon = []
    for key, value in coordinate_bounds.items():
        points = [
            (value["NW"]["coord"].x, value["NW"]["coord"].y),
            (value["NE"]["coord"].x, value["NE"]["coord"].y),
            (value["SE"]["coord"].x, value["SE"]["coord"].y),
            (value["SW"]["coord"].x, value["SW"]["coord"].y),
            (value["NW"]["coord"].x, value["NW"]["coord"].y)
        ]
        polygon.append(transform(project_toutm, Polygon(points)))
    for i in polygon:
        print(i.area)

    # Intersection of all polygons
    intersection = reduce(lambda p1, p2: p1.intersection(p2), polygon)

    # Calculate the area of the intersection
    intersection_area = intersection.area / 1000000
    print(f"The area shared by all polygons is: {intersection_area} km^2")
    print(intersection)

    # Transform back to WGS84
    intersection_wgs84 = transform(project_towgs84, intersection)
    print(intersection_wgs84)
    return intersection_wgs84


def prep_osm_roads(roads_path, proj_wgs84, intersection_wgs84):
    """ Filters the the OSM roadmap with the derived intersection area and saves the filtered roads for further usage. """
    roads_shapefile_path = f'{roads_path}/gis_osm_roads_free_1.shp'
    gdf_roads = gpd.read_file(roads_shapefile_path)

    AoI = Polygon(intersection_wgs84)
    gdf_road_intersection = gdf_roads[gdf_roads.intersects(AoI)]
    
    #filter roads by road code, 5141 are service and parking roads, therefore vehicles close to them are considered to be parking
    roads_gdf = roads_gdf[roads_gdf['code'] != 5141]
    
    #print(len(gdf_road_intersection))
    #print(gdf_road_intersection.head())
    
    gdf_road_intersection.to_file(f'{roads_path}/AoI_Roads.shp')
    print("OSM Roads within intersection saved.")