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
    for x in quality:
        im.save("automation/webp_" + str(x) + "/mri_" + str(i+1) + ".webp", CONST_WEBP, quality=x)
        im.save("automation/jpeg_" + str(x) + "/mri_" + str(i+1) + ".jpeg", CONST_JPEG, quality=x)


def get_paths(name, format=None, quality=None):
    files = glob.glob('automation/'+ name + '/*.bmp')
    if format == CONST_JPEG:
        files = glob.glob('automation/'+ name + '/*.jpeg')
    elif format == CONST_WEBP:
        files = glob.glob('automation/'+ name + '/*.webp')
    n = len(files)
    paths = []
    for i in range (n):
        if quality is None:
            paths.append("automation/" + name + "/mri_" + str(i+1) + ".bmp")
        else:
            if format == CONST_JPEG:
                paths.append("automation/" + name + "/mri_" + str(i+1) + ".jpeg")
            elif format == CONST_WEBP:
                paths.append("automation/" + name + "/mri_" + str(i+1) + ".webp")
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
        size_arr.append(f"{file_size_kb:.0f}")

    return psnr_arr, ssim_arr, size_arr
        
bmp_files = get_paths('bmp')
jpeg_100_files = get_paths('jpeg_100', CONST_JPEG, 100)
jpeg_90_files = get_paths('jpeg_90', CONST_JPEG, 90)
jpeg_80_files = get_paths('jpeg_80', CONST_JPEG, 80)
jpeg_70_files = get_paths('jpeg_70', CONST_JPEG, 70)
jpeg_60_files = get_paths('jpeg_60', CONST_JPEG, 60)
jpeg_50_files = get_paths('jpeg_50', CONST_JPEG, 50)
jpeg_40_files = get_paths('jpeg_40', CONST_JPEG, 40)
webp_40_files = get_paths('webp_40', CONST_WEBP,  40)
webp_50_files = get_paths('webp_50', CONST_WEBP, 50)
webp_60_files = get_paths('webp_60', CONST_WEBP, 60)
webp_70_files = get_paths('webp_70', CONST_WEBP, 70)
webp_80_files = get_paths('webp_80', CONST_WEBP, 80)
webp_90_files = get_paths('webp_90', CONST_WEBP, 90)
webp_100_files = get_paths('webp_100', CONST_WEBP, 100)


psnr_jpeg_100, ssim_jpeg_100, size_jpeg_100 = compare_psnr_ssim(bmp_files, jpeg_100_files)
psnr_jpeg_90, ssim_jpeg_90, size_jpeg_90 = compare_psnr_ssim(bmp_files, jpeg_90_files)
psnr_jpeg_80, ssim_jpeg_80, size_jpeg_80 = compare_psnr_ssim(bmp_files, jpeg_80_files)
psnr_jpeg_70, ssim_jpeg_70, size_jpeg_70 = compare_psnr_ssim(bmp_files, jpeg_70_files)
psnr_jpeg_60, ssim_jpeg_60, size_jpeg_60 = compare_psnr_ssim(bmp_files, jpeg_60_files)
psnr_jpeg_50, ssim_jpeg_50, size_jpeg_50 = compare_psnr_ssim(bmp_files, jpeg_50_files)
psnr_jpeg_40, ssim_jpeg_40, size_jpeg_40 = compare_psnr_ssim(bmp_files, jpeg_40_files)
psnr_40, ssim_40, size_40 = compare_psnr_ssim(bmp_files, webp_40_files)
psnr_50, ssim_50, size_50 = compare_psnr_ssim(bmp_files, webp_50_files)
psnr_60, ssim_60, size_60 = compare_psnr_ssim(bmp_files, webp_60_files)
psnr_70, ssim_70, size_70 = compare_psnr_ssim(bmp_files, webp_70_files)
psnr_80, ssim_80, size_80 = compare_psnr_ssim(bmp_files, webp_80_files)
psnr_90, ssim_90, size_90 = compare_psnr_ssim(bmp_files, webp_90_files)
psnr_100, ssim_100, size_100 = compare_psnr_ssim(bmp_files, webp_100_files)

seq = []
for i in range(pixel.shape[0]):
    seq.append(i+1)

data_jpeg_100 = {
    'Sequence': seq,
    'PSNR': psnr_jpeg_100,
    'SSIM': ssim_jpeg_100,
    'Size (KB)': size_jpeg_100,
}
data_jpeg_90 = {
    'Sequence': seq,
    'PSNR': psnr_jpeg_90,
    'SSIM': ssim_jpeg_90,
    'Size (KB)': size_jpeg_90,
}
data_jpeg_80 = {
    'Sequence': seq,
    'PSNR': psnr_jpeg_80,
    'SSIM': ssim_jpeg_80,
    'Size (KB)': size_jpeg_80,
}
data_jpeg_70 = {
    'Sequence': seq,
    'PSNR': psnr_jpeg_70,
    'SSIM': ssim_jpeg_70,
    'Size (KB)': size_jpeg_70,
}
data_jpeg_60 = {
    'Sequence': seq,
    'PSNR': psnr_jpeg_60,
    'SSIM': ssim_jpeg_60,
    'Size (KB)': size_jpeg_60,
}
data_jpeg_50 = {
    'Sequence': seq,
    'PSNR': psnr_jpeg_50,
    'SSIM': ssim_jpeg_50,
    'Size (KB)': size_jpeg_50,
}
data_jpeg_40 = {
    'Sequence': seq,
    'PSNR': psnr_jpeg_40,
    'SSIM': ssim_jpeg_40,
    'Size (KB)': size_jpeg_40,
}
data_40 = {
    'Sequence': seq,
    'PSNR': psnr_40,
    'SSIM': ssim_40,
    'Size (KB)': size_40,
}
data_50 = {
    'Sequence': seq,
    'PSNR': psnr_50,
    'SSIM': ssim_50,
    'Size (KB)': size_50,
}
data_60 = {
    'Sequence': seq,
    'PSNR': psnr_60,
    'SSIM': ssim_60,
    'Size (KB)': size_60,
}
data_70 = {
    'Sequence': seq,
    'PSNR': psnr_70,
    'SSIM': ssim_70,
    'Size (KB)': size_70,
}
data_80 = {
    'Sequence': seq,
    'PSNR': psnr_80,
    'SSIM': ssim_80,
    'Size (KB)': size_80,
}
data_90 = {
    'Sequence': seq,
    'PSNR': psnr_90,
    'SSIM': ssim_90,
    'Size (KB)': size_90,
}
data_100 = {
    'Sequence': seq,
    'PSNR': psnr_100,
    'SSIM': ssim_100,
    'Size (KB)': size_100,
}

# Creating DataFrames
df_jpeg_100 = pd.DataFrame(data_jpeg_100)
df_jpeg_90 = pd.DataFrame(data_jpeg_90)
df_jpeg_80 = pd.DataFrame(data_jpeg_80)
df_jpeg_70 = pd.DataFrame(data_jpeg_70)
df_jpeg_60 = pd.DataFrame(data_jpeg_60)
df_jpeg_50 = pd.DataFrame(data_jpeg_50)
df_jpeg_40 = pd.DataFrame(data_jpeg_40)
df_40 = pd.DataFrame(data_40)
df_50 = pd.DataFrame(data_50)
df_60 = pd.DataFrame(data_60)
df_70 = pd.DataFrame(data_70)
df_80 = pd.DataFrame(data_80)
df_90 = pd.DataFrame(data_90)
df_100 = pd.DataFrame(data_100)

# Specify the Excel file path
excel_file_path = 'automation/result.xlsx'

# Writing DataFrames to different sheets in the same Excel file
with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
    df_jpeg_100.to_excel(writer, sheet_name='JPEG 100', index=False)
    df_jpeg_90.to_excel(writer, sheet_name='JPEG 90', index=False)
    df_jpeg_80.to_excel(writer, sheet_name='JPEG 80', index=False)
    df_jpeg_70.to_excel(writer, sheet_name='JPEG 70', index=False)
    df_jpeg_60.to_excel(writer, sheet_name='JPEG 60', index=False)
    df_jpeg_50.to_excel(writer, sheet_name='JPEG 50', index=False)
    df_jpeg_40.to_excel(writer, sheet_name='JPEG 40', index=False)
    df_40.to_excel(writer, sheet_name='WEBP 40', index=False)
    df_50.to_excel(writer, sheet_name='WEBP 50', index=False)
    df_60.to_excel(writer, sheet_name='WEBP 60', index=False)
    df_70.to_excel(writer, sheet_name='WEBP 70', index=False)
    df_80.to_excel(writer, sheet_name='WEBP 80', index=False)
    df_90.to_excel(writer, sheet_name='WEBP 90', index=False)

df_jpeg_100['PSNR'] = pd.to_numeric(df_jpeg_100['PSNR'], errors='coerce')
df_jpeg_90['PSNR'] = pd.to_numeric(df_jpeg_90['PSNR'], errors='coerce')
df_jpeg_80['PSNR'] = pd.to_numeric(df_jpeg_80['PSNR'], errors='coerce')
df_jpeg_70['PSNR'] = pd.to_numeric(df_jpeg_70['PSNR'], errors='coerce')
df_jpeg_60['PSNR'] = pd.to_numeric(df_jpeg_60['PSNR'], errors='coerce')
df_jpeg_50['PSNR'] = pd.to_numeric(df_jpeg_50['PSNR'], errors='coerce')
df_jpeg_40['PSNR'] = pd.to_numeric(df_jpeg_40['PSNR'], errors='coerce')
df_40['PSNR'] = pd.to_numeric(df_40['PSNR'], errors='coerce')
df_50['PSNR'] = pd.to_numeric(df_50['PSNR'], errors='coerce')
df_60['PSNR'] = pd.to_numeric(df_60['PSNR'], errors='coerce')
df_70['PSNR'] = pd.to_numeric(df_70['PSNR'], errors='coerce')
df_80['PSNR'] = pd.to_numeric(df_80['PSNR'], errors='coerce')
df_90['PSNR'] = pd.to_numeric(df_90['PSNR'], errors='coerce')
df_100['PSNR'] = pd.to_numeric(df_100['PSNR'], errors='coerce')

df_jpeg_100['SSIM'] = pd.to_numeric(df_jpeg_100['SSIM'], errors='coerce')
df_jpeg_90['SSIM'] = pd.to_numeric(df_jpeg_90['SSIM'], errors='coerce')
df_jpeg_80['SSIM'] = pd.to_numeric(df_jpeg_80['SSIM'], errors='coerce')
df_jpeg_70['SSIM'] = pd.to_numeric(df_jpeg_70['SSIM'], errors='coerce')
df_jpeg_60['SSIM'] = pd.to_numeric(df_jpeg_60['SSIM'], errors='coerce')
df_jpeg_50['SSIM'] = pd.to_numeric(df_jpeg_50['SSIM'], errors='coerce')
df_jpeg_40['SSIM'] = pd.to_numeric(df_jpeg_40['SSIM'], errors='coerce')
df_40['SSIM'] = pd.to_numeric(df_40['SSIM'], errors='coerce')
df_50['SSIM'] = pd.to_numeric(df_50['SSIM'], errors='coerce')
df_60['SSIM'] = pd.to_numeric(df_60['SSIM'], errors='coerce')
df_70['SSIM'] = pd.to_numeric(df_70['SSIM'], errors='coerce')
df_80['SSIM'] = pd.to_numeric(df_80['SSIM'], errors='coerce')
df_90['SSIM'] = pd.to_numeric(df_90['SSIM'], errors='coerce')
df_100['SSIM'] = pd.to_numeric(df_100['SSIM'], errors='coerce')

plt.figure(figsize=(12, 6))
plt.plot(seq, df_jpeg_40['PSNR'], label='JPEG 40', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_50['PSNR'], label='JPEG 50', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_60['PSNR'], label='JPEG 60', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_70['PSNR'], label='JPEG 70', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_80['PSNR'], label='JPEG 80', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_90['PSNR'], label='JPEG 90', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_100['PSNR'], label='JPEG 100', marker='o', linestyle='-')
plt.plot(seq, df_40['PSNR'], label='WEBP 40', marker='o', linestyle='-')
plt.plot(seq, df_50['PSNR'], label='WEBP 50', marker='o', linestyle='-')
plt.plot(seq, df_60['PSNR'], label='WEBP 60', marker='o', linestyle='-')
plt.plot(seq, df_70['PSNR'], label='WEBP 70', marker='o', linestyle='-')
plt.plot(seq, df_80['PSNR'], label='WEBP 80', marker='o', linestyle='-')
plt.plot(seq, df_90['PSNR'], label='WEBP 90', marker='o', linestyle='-')
plt.plot(seq, df_100['PSNR'], label='WEBP 100', marker='o', linestyle='-')
plt.title('PSNR Comparison')
plt.xlabel('Sequence')
plt.ylabel('PSNR')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('automation/plot/psnr.png')

plt.figure(figsize=(12, 6))
plt.plot(seq, df_jpeg_40['SSIM'], label='JPEG 40', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_50['SSIM'], label='JPEG 50', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_60['SSIM'], label='JPEG 60', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_70['SSIM'], labsel='JPEG 70', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_80['SSIM'], label='JPEG 80', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_90['SSIM'], label='JPEG 90', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_100['SSIM'], label='JPEG 100', marker='o', linestyle='-')
plt.plot(seq, df_40['SSIM'], label='WEBP 40', marker='o', linestyle='-')
plt.plot(seq, df_50['SSIM'], label='WEBP 50', marker='o', linestyle='-')
plt.plot(seq, df_60['SSIM'], label='WEBP 60', marker='o', linestyle='-')
plt.plot(seq, df_70['SSIM'], label='WEBP 70', marker='o', linestyle='-')
plt.plot(seq, df_80['SSIM'], label='WEBP 80', marker='o', linestyle='-')
plt.plot(seq, df_90['SSIM'], label='WEBP 90', marker='o', linestyle='-')
plt.plot(seq, df_100['SSIM'], label='WEBP 100', marker='o', linestyle='-')
plt.title('SSIM Comparison')
plt.xlabel('Sequence')
plt.ylabel('SSIM')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('automation/plot/ssim.png')

plt.figure(figsize=(12, 6))
plt.plot(seq, df_jpeg_40['Size (KB)'], label='JPEG 40', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_50['Size (KB)'], label='JPEG 50', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_60['Size (KB)'], label='JPEG 60', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_70['Size (KB)'], label='JPEG 70', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_80['Size (KB)'], label='JPEG 80', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_90['Size (KB)'], label='JPEG 90', marker='o', linestyle='-')
plt.plot(seq, df_jpeg_100['Size (KB)'], label='JPEG 100', marker='o', linestyle='-')
plt.plot(seq, df_40['Size (KB)'], label='WEBP 40', marker='o', linestyle='-')
plt.plot(seq, df_50['Size (KB)'], label='WEBP 50', marker='o', linestyle='-')
plt.plot(seq, df_60['Size (KB)'], label='WEBP 60', marker='o', linestyle='-')
plt.plot(seq, df_70['Size (KB)'], label='WEBP 70', marker='o', linestyle='-')
plt.plot(seq, df_80['Size (KB)'], label='WEBP 80', marker='o', linestyle='-')
plt.plot(seq, df_90['Size (KB)'], label='WEBP 90', marker='o', linestyle='-')
plt.plot(seq, df_100['Size (KB)'], label='WEBP 100', marker='o', linestyle='-')
plt.title('Size Comparison')
plt.xlabel('Sequence')
plt.ylabel('Size')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('automation/plot/size.png')