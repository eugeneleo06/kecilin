import pydicom
import numpy as np
from PIL import Image
import glob
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import matplotlib.pyplot as plt
import imagecodecs

CONST_WEBP = '.WEBP'
CONST_JXL = '.JXL'
CONST_JP2 = '.JP2'

quality = [100,90,80,70,60,50,40,30,20,10]
ext = [CONST_WEBP, CONST_JXL, CONST_JP2]

folder = "automation_brain2/"

def generate_key(extension, quality):
    return f"{extension[1:].lower()}_{quality}"

for x in ext:
    os.makedirs("upload/" + folder + x[1:].lower(), exist_ok=True)
    for y in quality:
        key = generate_key(x,y)
        save_folder = "upload/" + folder + x[1:].lower() + "/" + key
        os.makedirs(save_folder, exist_ok=True)
        files = glob.glob(folder + key + '/*' + x.lower())
        n = len(files)
        path = []
        for i in range (n):
            path.append(folder + key + "/mri_" + str(i+1) + x.lower())
        i=0
        for p in path:
            if x == CONST_JXL:
                img_comp = imagecodecs.imread(p)
            else:
                img_comp = cv2.imread(p)
            print(p)
            im = Image.fromarray(img_comp)
            im.save(save_folder + "/mri_" + str(i+1) + ".bmp", 'BMP')
            i += 1