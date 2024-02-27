import pydicom
import numpy as np
from PIL import Image
import glob
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

quality = [90,80,70,60,50,40]

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
    im.save("automation/jpeg/mri_" + str(i+1) + ".jpeg", 'JPEG')

    for x in quality:
        im.save("automation/webp_" + str(x) + "/mri_" + str(i+1) + ".webp", 'WEBP', quality=x)


def get_paths(format, quality=None):
    files = glob.glob('automation/'+ format + '/*.' + format)
    if quality is not None:
        files = glob.glob('automation/'+ format + '/*.webp')
    n = len(files)
    paths = []
    for i in range (n):
        if quality is not None:
            paths.append("automation/" + format + "/mri_" + str(i+1) + ".webp")
        else:
            paths.append("automation/" + format + "/mri_" + str(i+1) + "." + format)
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
        size_arr.append(f"{file_size_kb:.4f}")

    return psnr_arr, ssim_arr, size_arr
        
bmp_files = get_paths('bmp')
jpeg_files = get_paths('jpeg')
webp_40_files = get_paths('webp_40', 40)
webp_50_files = get_paths('webp_50', 50)
webp_60_files = get_paths('webp_60', 60)
webp_70_files = get_paths('webp_70', 70)
webp_80_files = get_paths('webp_80', 80)
webp_90_files = get_paths('webp_90', 90)


psnr_jpeg, ssim_jpeg, size_jpeg = compare_psnr_ssim(bmp_files, jpeg_files)
psnr_40, ssim_40, size_40 = compare_psnr_ssim(bmp_files, webp_40_files)
psnr_50, ssim_50, size_50 = compare_psnr_ssim(bmp_files, webp_50_files)
psnr_60, ssim_60, size_60 = compare_psnr_ssim(bmp_files, webp_60_files)
psnr_70, ssim_70, size_70 = compare_psnr_ssim(bmp_files, webp_70_files)
psnr_80, ssim_80, size_80 = compare_psnr_ssim(bmp_files, webp_80_files)
psnr_90, ssim_90, size_90 = compare_psnr_ssim(bmp_files, webp_90_files)


seq = []
for i in range(pixel.shape[0]):
    seq.append(i+1)

data_jpeg = {
    'Sequence': seq,
    'PSNR': psnr_jpeg,
    'SSIM': ssim_jpeg,
    'Size (KB)': size_jpeg,
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

# Creating DataFrames
df_jpeg = pd.DataFrame(data_jpeg)
df_40 = pd.DataFrame(data_40)
df_50 = pd.DataFrame(data_50)
df_60 = pd.DataFrame(data_60)
df_70 = pd.DataFrame(data_70)
df_80 = pd.DataFrame(data_80)
df_90 = pd.DataFrame(data_90)

# Specify the Excel file path
excel_file_path = 'automation/result.xlsx'

# Writing DataFrames to different sheets in the same Excel file
with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
    df_jpeg.to_excel(writer, sheet_name='JPEG', index=False)
    df_40.to_excel(writer, sheet_name='WEBP 40', index=False)
    df_50.to_excel(writer, sheet_name='WEBP 50', index=False)
    df_60.to_excel(writer, sheet_name='WEBP 60', index=False)
    df_70.to_excel(writer, sheet_name='WEBP 70', index=False)
    df_80.to_excel(writer, sheet_name='WEBP 80', index=False)
    df_90.to_excel(writer, sheet_name='WEBP 90', index=False)
