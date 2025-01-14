##only runs on windows with gdal

import numpy as np
import rasterio
import rasterio.plot
from PIL import Image
import os
import cv2

def create_folders(path):
    """ 
    Takes the names of folders and images as manual input and returns list of strings.
    """
    
    folders = [path + "001",
                path + "002",
                path + "003",
                path + "004", 
                path + "005", 
                path + "006"
            ]
    images =    [["001_Part1.TIF", "001_Part2.TIF"],
                ["002_Part1.TIF", "002_Part2.TIF"],
                ["003_Part1.TIF", "003_Part2.TIF"],
                ["004_Part1.TIF", "004_Part2.TIF", "004_Part3.TIF"],
                ["005_Part1.TIF", "005_Part2.TIF", "005_Part3.TIF", "005_Part4.TIF"],
                ["006_Part1.TIF", "006_Part2.TIF", "006_Part3.TIF"]
            ]
    return folders, images

def split_into_patches(array, patch_size, path):
    """
    Split an array into smaller square patches. 
    
    Parameters:
        array (ndarray): The input array.
        patch_size (int): The size of each square patch.
    
    Returns:
        Saves splitted images.
    """
    patches = []
    height, width = array.shape[1:]
    print(height, width, array.shape)
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            #print(y, x, y+patch_size, x+patch_size)
            patch = array[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            #print(patch.shape)
            arr = np.ascontiguousarray(patch.transpose(1,2,0))
            img = Image.fromarray(arr, 'RGB')
            img.save(f'{path}/dist/splitted_raw{i[:-4]}_{patch_size}_{y}_{x}.png')
            #rasterio.plot.show(patch)

def correct_color_and_split(path, patch_size):
    """
    Loops throughs folders and for each image it corrects the color and splits it. 
    """
    folders, images = create_folders()
    for folder in folders:
        for image in images:
            for i in image:
                image_path = os.path.join(folder, i)
                try:
                    tif_image = rasterio.open(path)
                    print(image_path + " loaded")
                except:
                    print(image_path + " failed")
                    continue
                    
                image = tif_image.read([1,2,3]) #read rgb
                #alpha = tif_image.read([4])     #read alpha
                image = np.round(np.power(image, 0.5)).astype(image.dtype)  #take the sqrt to preserve usable scaling
                image = (255 * image / np.max(image)).astype(np.uint8)      #rescale to 255 uint8 max
                
                # Adjust the brightness and contrast
                brightness = 10 
                contrast = 1.5
                image = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness) 
                #rasterio.plot.show(image)

                #Split the array into smaller patches of size 640x640
                split_into_patches(image, patch_size, image_path)
