#%%
import os
import geopandas as gpd
from PIL import Image
import datetime
from shapely.geometry import LineString


def predict_loop(path, version30cm, version50cm, version50cmsnow, iou, max_det, name_codes, coordinate_bounds):
    current_datetime = datetime.datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


    gdf = gpd.DataFrame(columns=["box", "confidence", "labels", "image_code", "coordinates"])
    for root, dirs, files in os.walk(f'{path}/dist'):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file_name in files:
            try:
                file_path = os.path.join(root, file_name)
                #print("File:", file_path)
                image = Image.open(file_path)
                if "PNEO4" in file_name:
                    satellite = "PNEO4"
                    results = version30cm(image, device='mps', conf = name_codes[file_name]["max_conf"], iou = iou, max_det = max_det)
                elif "PHR1A" in file_name:
                    satellite = "PHR1A"
                    results = version50cmsnow(image, device='mps', conf = name_codes[file_name]["max_conf"], iou = iou, max_det = max_det)
                elif "PHR1B" in file_name:
                    satellite = "PHR1B"
                    results = version50cm(image, device='mps', conf = name_codes[file_name]["max_conf"], iou = iou, max_det = max_det)
                else:
                    print("weird image -> check name")
                    break
                #print(file_name)
                #print(file_path)
                #print(results)
                annotated_image = results[0].plot(labels= False, conf = False)
                im = Image.fromarray(annotated_image[..., ::-1])  # RGB PIL image
                #im.show()  # show image

                code = name_codes[file_name]['code']
                parts = name_codes[file_name]['parts']
                #print(file_name) 
                """ 
                The information about the position of the image chunk within the original image was transported in the 
                file name. Hence the following code. Position of split needs to be adapted to the specific images available.
                """
                tile = int(file_name.split("_")[-4].replace("C1", "").replace("R", ""))
                #print(tile)
                row = int(file_name.split("_")[-2:-1][0])  + (tile - 1) * coordinate_bounds[code]['split']
                #print(row)
                col = int(file_name.split("_")[-1].replace(".png",""))
                #print(col)
                if satellite == "PNEO4":
                    split_no = file_name[:-4] #depends on filename
                else:
                    split_no = file_name[:-4] #depends on filename
                print(split_no)
                    
                im.save(f'{path}/dist/result_images/res_{satellite}_{code}_{split_no}_{current_datetime_str}.png')
                
                ### Count detections
                #print(results)
                image_code, coordinate = [], []
                for pred in results:
                    box_list = pred.boxes.xyxy.tolist()
                    if box_list != []:
                        for box in box_list:
                            center = {"col": (box[1] + box[3]) / 2, "row": (box[0] + box[2]) / 2}
                            percent_col = (float(center["col"] + col) / float(coordinate_bounds[code]["SE"]["col"]))
                            percent_row = (float(center["row"] + row) / float(coordinate_bounds[code]["SE"]["row"]))
                            coordinate.append(get_coordinate_of_object(coordinate_bounds[code]["NW"]["coord"], coordinate_bounds[code]["NE"]["coord"], coordinate_bounds[code]["SE"]["coord"], coordinate_bounds[code]["SW"]["coord"], percent_col, percent_row))

                    confs = pred.boxes.conf.tolist()
                    labels = pred.boxes.cls.tolist()

                image_code = [code for _ in range(len(box_list))]
                gdf_temp = gpd.DataFrame({"box": box_list, "confidence": confs, "labels": labels, "image_code": image_code, "coordinates": coordinate})
                gdf = gpd.concat([gdf, gdf_temp], ignore_index=True)
            except:
                print(f"{file_path} failed")
            

    gdf.to_file(f'{path}/dist/res_{current_datetime_str}.shp')


def get_coordinate_of_object(coordNW, coordNE, coordSE, coordSW, percent_col, percent_row):
    """
    Helper function to convert the position of the box in the image into coordinates
    
    Returns:
        x and y coordinate of detected object
    """
    # Create a LineString object from the two coordinates
    NW_to_NE = LineString([coordNW, coordNE])
    SW_to_SE = LineString([coordSW, coordSE])

    
    # Interpolate a point at the specified percentage along the line
    NW_to_NE_percent = NW_to_NE.interpolate(percent_col, normalized=True)
    SW_to_SE_percent = SW_to_SE.interpolate(percent_col, normalized=True)

    NW_to_NE_percent__to__SW_to_SE_percent = LineString([NW_to_NE_percent, SW_to_SE_percent]) 

    coord_of_object = NW_to_NE_percent__to__SW_to_SE_percent.interpolate(percent_row, normalized=True)
    return (coord_of_object.x, coord_of_object.y)