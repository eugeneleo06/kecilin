import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
import json
import cv2
import glob
import numpy as np
from PIL import Image
import io

# Opening JSON file
f = open('automation_brain3/mri/meta.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)

# Convert JSON data back to dictionary
json_data_dict_read = json.loads(data)

# Remove DICOM File Meta Information from dictionary
meta_info_read = json_data_dict_read.pop('_meta_info', None)

isImplicitVR = eval(json_data_dict_read.pop('_is_implicit_vr', None))
isLittleEndian = eval(json_data_dict_read.pop('_is_little_endian', None))


# Convert dictionary back to JSON string
data_without_meta = json.dumps(json_data_dict_read)

ds = Dataset.from_json(data_without_meta)

ds.file_meta = Dataset.from_json(json.dumps(meta_info_read))

webp_files = glob.glob('automation_brain3/mri/*.webp')

n = len(webp_files)

webp_paths = []

for i in range (n):
    webp_paths.append('automation_brain3/mri/mri_'+ str(i) +'.webp')

print(webp_paths)

jpeg_frames_as_bytes = []
for file_path in webp_paths:
    # Read WebP image using cv2
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imencode returns a tuple where the first element is a success flag and the second is the buffer
    pil_img = Image.fromarray(img)
    
    with io.BytesIO() as output:
        pil_img.save(output, format="JPEG")
        jpeg_frames_as_bytes.append(output.getvalue())

# Encapsulate the JPEG byte data for DICOM
ds.PixelData = pydicom.encaps.encapsulate(jpeg_frames_as_bytes)

ds.is_implicit_VR = isImplicitVR
ds.is_little_endian = isLittleEndian
ds.preamble = b'\0' * 128

ds.save_as('automation_brain3/mri/mri_res.dcm')