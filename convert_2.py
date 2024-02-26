import cv2
import pydicom
import os
import numpy as np
from skimage import exposure
from pydicom.pixel_data_handlers.util import apply_color_lut, convert_color_space
from PIL import Image

im = pydicom.dcmread("mri.dcm")
# print(im)
im = im.pixel_array
print(np.shape(im))

for i in range(im.shape[0]):
    img = im[i].astype(float)

    rscl_img = (np.maximum(img, 0) / img.max())*255
    final_img = np.uint8(rscl_img)
    cv2.imwrite("mri1/mri_" + str(i)+".webp", final_img, [int(cv2.IMWRITE_WEBP_QUALITY), 90])

# cv2.imwrite("mri2.webp", im, [int(cv2.IMWRITE_WEBP_QUALITY), 90])
# cv2.imwrite("mri2_scaled.webp", final_img, [int(cv2.IMWRITE_WEBP_QUALITY), 90])