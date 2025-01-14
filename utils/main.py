from dotenv import load_dotenv
import os
import Roboflow
import pyproj
from shapely.geometry import Point


load_dotenv()

# %%
######################################################
#### Input Variables #################################
######################################################
patch_size = 640

### Projection
proj_wgs84 = pyproj.CRS('EPSG:4326')
proj_utm = pyproj.CRS('EPSG:32636')
crs_Kyiv = 9864


#### .env
path = os.getenv("PATH")
train_path = f"{path}/dist/models/runs/detect"
folder_path = f'{path}/dist'
roads_path = f'{path}/src/ukr_roads/'
figures_path = f'{path}/dist/figures'


roboflow_key = os.getenv("ROBOKEY")
roboflow_workspace = os.getenv("ROBOWORKSPACE")
roboflow_project = os.getenv("ROBOPROJECT")
rf = Roboflow(api_key=roboflow_key)
project = rf.workspace(roboflow_workspace).project(roboflow_workspace)

#### model and roboflow project
model_type = "yolov11"

#Project Versions of resolution samples
version30cm = 1
version50cm = 2
version50cmsnow = 3

#Project Versions of single images from Roboflow
v20230928 = 21
v20230210 = 20
v20220415 = 19
v20220325 = 18
v20220204 = 17
v20191206 = 16
v_image_list = [v20230928, v20230210, v20220415, v20220325, v20220204, v20191206]

# Load a model
model_30cm= YOLO(f'{path}/dist/models/runs/detect/train{version30cm}/weights/best.pt')
model_50cm = YOLO(f'{path}/dist/models/runs/detect/train{version50cm}/weights/best.pt')
model_50cm_snow = YOLO(f'{path}/dist/models/runs/detect/train{version50cmsnow}/weights/best.pt')

#model specifications
iou = 0.5
max_det = 10000
distance_to_road = 5

##future update: automatically extract information from xml files
name_codes = {"PNEO4_202309280908597":   {"code": "1", "date": "202309280908597", "parts": 4, "max_conf": None}, 
              "PNEO4_202204150857427":   {"code": "2", "date": "202204150857427", "parts": 3, "max_conf": None},
              "PNEO4_202203250853461":   {"code": "3", "date": "202203250853461", "parts": 3, "max_conf": None},
              "PHR1A_202202040916288":   {"code": "4", "date": "202202040916288", "parts": 2, "max_conf": None},
              "PHR1B_201912060900436":   {"code": "5", "date": "201912060900436", "parts": 2, "max_conf": None},
              "PHR1A_202302100911426":   {"code": "6", "date": "202302100911426", "parts": 2, "max_conf": None},
}

coordinate_bounds = {"1":  {"NW": {"coord": Point(30.441895751663836,   50.52035336713138),    "col": "1",        "row": "1"}, 
                            "NE": {"coord": Point(30.48070230421028,    50.52120004086682),    "col": "9180",     "row": "1"}, 
                            "SE": {"coord": Point(30.487142311059312,   50.39988830049867),    "col": "9180",     "row": "45008"}, 
                            "SW": {"coord": Point(30.4484346977782,     50.3990452467174),     "col": "1",        "row": "45008"},
                            "split": 14336},
                    "2":   {"NW": {"coord": Point(30.443614666515735,   50.491679196130775),   "col": "1",        "row": "1"}, 
                            "NE": {"coord": Point(30.480741459780745,   50.49248910089108),    "col": "8788",     "row": "1"}, 
                            "SE": {"coord": Point(30.485455270581916,   50.40368371563942),    "col": "8788",     "row": "32948"}, 
                            "SW": {"coord": Point(30.44839776301755,    50.40287634699985),    "col": "1",        "row": "32948"},
                            "split": 14336}, 
                    "3":   {"NW": {"coord": Point(30.444770077278918,   50.51259091083697),    "col": "1",        "row": "1"}, 
                            "NE": {"coord": Point(30.48078042377268,    50.51337616239662),    "col": "8520",     "row": "1"}, 
                            "SE": {"coord": Point(30.486724505229926,   50.4014012314574),     "col": "8520",     "row": "41544"}, 
                            "SW": {"coord": Point(30.45079890185057,    50.40061907928971),    "col": "1",        "row": "41544"},
                            "split": 14336}, 
                    "4":   {"NW": {"coord": Point(30.43680050240526,    50.51082125420503),    "col": "1",        "row": "1"}, 
                            "NE": {"coord": Point(30.48638220337783,    50.51190296378905),    "col": "7039",     "row": "1"}, 
                            "SE": {"coord": Point(30.49263389478426,    50.39384390573065),    "col": "7039",     "row": "26281"}, 
                            "SW": {"coord": Point(30.44317518506564,    50.39276669699363),    "col": "1",        "row": "26281"},
                            "split": 16384}, 
                    "5":   {"NW": {"coord": Point(30.43595414290481,    50.52643585642573),    "col": "1",        "row": "1"}, 
                            "NE": {"coord": Point(30.48601730112088,    50.5275282125948),     "col": "7105",     "row": "1"}, 
                            "SE": {"coord": Point(30.49309771615375,    50.39385390799088),    "col": "7105",     "row": "29757"}, 
                            "SW": {"coord": Point(30.44317518506564,    50.39276669699363),    "col": "1",        "row": "29757"},
                            "split": 16384}, 
                    "6":   {"NW": {"coord": Point(30.43595414290481,    50.52643585642573),    "col": "1",        "row": "1"}, 
                            "NE": {"coord": Point(30.48601730112088,    50.5275282125948),     "col": "7105",     "row": "1"}, 
                            "SE": {"coord": Point(30.49309771615375,    50.39385390799088),    "col": "7105",     "row": "29757"}, 
                            "SW": {"coord": Point(30.44317518506564,    50.39276669699363),    "col": "1",        "row": "29757"},
                            "split": 16384}, 
}

# %%
######################################################
#### Preprocessing ###################################
######################################################
from aPre.splitting import * 
from aPre.osm_roads import *

# Image Processing / Splitting
correct_color_and_split(path, patch_size)

# Roads Processing
intersection_wgs84 = intersection(coordinate_bounds, proj_wgs84, proj_utm)
prep_osm_roads(roads_path, proj_wgs84, intersection_wgs84)

# %%
######################################################
#### Model Training ##################################
######################################################
from bTrain.model_training import *


#Call Functions
train_models(project, train_path, version30cm, version50cm, version50cmsnow, model_type)

upload_weights(project, train_path, version30cm, version50cm, version50cmsnow, model_type, v_image_list)

name_codes = validate(train_path, version30cm, version50cm, version50cmsnow)

# %%
######################################################
#### Predictions #####################################
######################################################
from cPredictions.prediction import * 

predict_loop(path, version30cm, version50cm, version50cmsnow, iou, max_det, name_codes, coordinate_bounds)

# %%
######################################################
#### Post Processing #################################
######################################################
from dPost.parking import * 
from dPost.plotting import *

observations_gdf = parking_vehicles(roads_path, path, intersection_wgs84, crs_Kyiv, distance_to_road)

create_grouped_results(observations_gdf)

nbins=100
roads_gdf = gpd.read_file(f'{roads_path}/AoI_Roads.shp')

#plot_obs_roads(observations_gdf, roads_gdf, figures_path)
#plot_Hist2D(observations_gdf, roads_gdf, figures_path, nbins)
#plot_KDE(observations_gdf, roads_gdf, figures_path, nbins)
#plot_intersection_area(roads_path, figures_path, intersection_wgs84)