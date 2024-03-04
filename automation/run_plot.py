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
import plotly.graph_objects as go

CONST_JPEG = 'JPEG'
CONST_WEBP = 'WEBP'

quality = [100,90,80,70,60,50,40]
ext = ['.JPEG', '.WEBP']

# Path to your single DICOM file
file_path = 'automation/mri.dcm'

# Read the DICOM file
dicom = pydicom.dcmread(file_path)

pixel = dicom.pixel_array

bmp_imgs = []
for i in range(pixel.shape[0]):
    img = pixel[i].astype(float)

    rscl_img = (np.maximum(img, 0) / img.max()) * 255
    final_img = np.uint8(rscl_img)

    im = Image.fromarray(final_img)
    im.save("automation/bmp/mri_" + str(i+1) + ".bmp", 'BMP')
    for x in ext:
        for y in quality:
            im.save("automation/" + x[1:].lower() +"_" + str(y) + "/mri_" + str(i+1) + x.lower() , x[1:], quality=y)

def get_paths(name: str, format: str = None):
    # name = jpeg_90
    # format = .jpeg
    if name == "bmp":
        files = glob.glob('automation/'+ name + '/*.bmp')
    else:
        files = glob.glob('automation/'+ name + '/*' + format.lower())
    n = len(files)
    paths = []
    for i in range (n):
        if name == "bmp":
            paths.append("automation/" + name + "/mri_" + str(i+1) + ".bmp")
        else:
            paths.append("automation/" + name + "/mri_" + str(i+1) + format.lower())
    return paths

def compare_psnr_ssim(bmp_files, files):
    psnr_arr = []
    ssim_arr = []
    size_arr = []

    n = len(bmp_files)

    for x in range (n):
        img_ori = cv2.imread(bmp_files[x])
        img_comp = cv2.imread(files[x])
        

        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        img_comp = cv2.cvtColor(img_comp, cv2.COLOR_BGR2GRAY)

        img_ori_arr = np.asarray(img_ori, dtype=np.float32)
        img_comp_arr = np.asarray(img_comp, dtype=np.float32)

        psnr_value = psnr(img_ori_arr, img_comp_arr, data_range=255)
        ssim_value, _ = ssim(img_ori_arr, img_comp_arr, full=True, data_range=255)

        # Get the file size in bytes
        file_size_bytes = os.path.getsize(files[x])

        # Convert to kilobytes (1 KB = 1024 bytes)
        file_size_kb = file_size_bytes / 1024
        
        psnr_arr.append(f"{psnr_value:.4f}")
        ssim_arr.append(f"{ssim_value:.5f}")
        size_arr.append(f"{file_size_kb:.2f}")

    return psnr_arr, ssim_arr, size_arr

def generate_key(extension, quality):
    return f"{extension[1:].lower()}_{quality}"

data, df, seq = {}, {}, []
files, psnrData, ssimData, sizeData = {}, {}, {}, {}
for i in range(pixel.shape[0]):
    seq.append(i+1)

files["bmp"] = get_paths('bmp')

for x in ext:
    for y in quality:
        files[generate_key(x,y)] = get_paths(generate_key(x,y), x)
        psnrData[generate_key(x,y)], ssimData[generate_key(x,y)], sizeData[generate_key(x,y)] = compare_psnr_ssim(files['bmp'], files[generate_key(x,y)])
        data[generate_key(x,y)] = {
            'Sequence': seq,
            'PSNR': psnrData[generate_key(x,y)],
            'SSIM': ssimData[generate_key(x,y)],
            'Size (KB)': sizeData[generate_key(x,y)],
        }
        df[generate_key(x,y)] = pd.DataFrame(data[generate_key(x,y)])

# Specify the Excel file path
excel_file_path = 'automation/result.xlsx'

# Writing DataFrames to different sheets in the same Excel file
with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
    for x in ext:
        for y in quality:
            df[generate_key(x,y)].to_excel(writer, sheet_name=generate_key(x,y), index=False)
            df[generate_key(x,y)]['PSNR'] = pd.to_numeric(df[generate_key(x,y)]['PSNR'], errors='coerce')
            df[generate_key(x,y)]['SSIM'] = pd.to_numeric(df[generate_key(x,y)]['SSIM'], errors='coerce')
            df[generate_key(x,y)]['Size (KB)'] = pd.to_numeric(df[generate_key(x,y)]['Size (KB)'], errors='coerce')

plt.figure(figsize=(12, 6))
for x in ext:
    for y in quality:
        plt.plot(seq, df[generate_key(x,y)]['PSNR'], label=generate_key(x,y), linestyle='-')
plt.title('PSNR Comparison')
plt.xlabel('Sequence')
plt.ylabel('PSNR')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('automation/plot/psnr.png')

plt.figure(figsize=(12, 6))
for x in ext:
    for y in quality:
        plt.plot(seq, df[generate_key(x,y)]['SSIM'], label=generate_key(x,y), linestyle='-')
plt.title('SSIM Comparison')
plt.xlabel('Sequence')
plt.ylabel('SSIM')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('automation/plot/ssim.png')

plt.figure(figsize=(12, 6))
for x in ext:
    for y in quality:
        plt.plot(seq, df[generate_key(x,y)]['Size (KB)'], label=generate_key(x,y), linestyle='-')
plt.title('Size Comparison')
plt.xlabel('Sequence')
plt.ylabel('Size (KB)')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('automation/plot/size.png')