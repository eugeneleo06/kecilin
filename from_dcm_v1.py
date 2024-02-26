import cv2
import pydicom
import os
import numpy as np
from pydicom.dataset import FileDataset
from skimage import exposure
from pydicom.pixel_data_handlers.util import apply_color_lut, convert_color_space
from PIL import Image
import pandas as pd 
import os

# MULTIPLE IMAGES IN ONE FOLDER

def count_files_in_dir(directory):
    if not os.path.isdir(directory):
        return "Error: Not a valid directory"
    
    file_count = 0
    for _, _, files in os.walk(directory):
        file_count += len(files)
    
    return file_count

directory_path = "Dataset/series-1"
n = count_files_in_dir(directory_path)


for i in range (n):

    # CONVERT TO WEBP
    filename = "{:06}".format(i+1) 
    im = pydicom.dcmread("Dataset/series-1/image-" + filename +".dcm")
    print(im)
    im = im.pixel_array
    img = im.astype(float)
    rscl_img = (np.maximum(img, 0) / img.max())*255
    final_img = np.uint8(rscl_img)
    cv2.imwrite("series-1/image-" + filename + ".webp", final_img, [int(cv2.IMWRITE_WEBP_QUALITY), 90])

    #STORE METADATA

