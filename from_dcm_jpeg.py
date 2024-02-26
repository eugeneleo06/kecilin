import cv2
import pydicom
import numpy as np
import json

# Path to your single DICOM file
file_path = 'mri_single_jpeg/mri.dcm'

# Read the DICOM file
obj = pydicom.dcmread(file_path)

obj1 = obj.pixel_array

for i in range(obj1.shape[0]):
    img = obj1[i].astype(float)

    # Normalize the image to 0-255
    rscl_img = (np.maximum(img, 0) / img.max()) * 255
    final_img = np.uint8(rscl_img)

    # Save the image as JPEG with specified quality
    cv2.imwrite("mri_single_jpeg/mri_" + str(i) + ".jpeg", final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

del obj.PixelData

json_data = obj.to_json()
meta_info = obj.file_meta.to_json()

# Convert JSON data to a dictionary
json_data_dict = json.loads(json_data)

# Add DICOM File Meta Information to dictionary
json_data_dict['_meta_info'] = json.loads(meta_info)

json_data_dict['_is_implicit_vr'] = str(obj.is_implicit_VR)

json_data_dict['_is_little_endian'] = str(obj.is_little_endian)

# Convert dictionary back to JSON string
json_data_with_meta = json.dumps(json_data_dict)

with open('mri_single_jpeg/meta.json', 'w') as outfile:
    json.dump(json_data_with_meta, outfile)