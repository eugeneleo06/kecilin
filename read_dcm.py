import cv2
import pydicom
import numpy as np
import json
import matplotlib.pyplot as plt

# Path to your single DICOM file
file_path = 'mri_single_jpeg/mri_res.dcm'

# file_path = 'Dataset/series-1/image-000001.dcm'


# Read the DICOM file
obj = pydicom.dcmread(file_path)

print(obj)

# plt.imshow(obj.pixel_array, cmap=plt.cm.gray)
# plt.show()
