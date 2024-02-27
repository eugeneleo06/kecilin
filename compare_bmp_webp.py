import cv2
import pydicom
import numpy as np
import json
import os
from skimage.metrics import structural_similarity as ssim

# Path to your single DICOM file
file_path = 'mri.dcm'

# Read the DICOM file
obj = pydicom.dcmread(file_path)

obj1 = obj.pixel_array
folder_path_1 = "mri_bmp"
folder_path_2 = "mri_webp"

def convert(image, filepath, method, quality):
    if method == "bmp":
        cv2.imwrite("mri_bmp/mri_" + str(i) + ".bmp", final_img)
            
    elif method == "webp":
        cv2.imwrite("mri_webp/mri_" + str(i) + ".webp", final_img, [int(cv2.IMWRITE_WEBP_QUALITY), 90])

# Fungsi untuk menghitung PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    # Mengkonversi gambar ke grayscale karena SSIM biasanya dihitung pada gambar grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Menghitung SSIM antara dua gambar grayscale
    ssim_value = ssim(img1_gray, img2_gray, data_range=img1_gray.max() - img1_gray.min())
    
    return ssim_value

for i in range(obj1.shape[0]):
    img = obj1[i].astype(float)

    rscl_img = (np.maximum(img, 0) / img.max()) * 255
    final_img = np.uint8(rscl_img)
    convert(final_img, obj1, "bmp", 90)
    convert(final_img, obj1, "webp", 90)


# Mendapatkan daftar file dari kedua folder dan mengurutkannya
files_folder_1 = sorted(os.listdir(folder_path_1))
files_folder_2 = sorted(os.listdir(folder_path_2))

# Pastikan jumlah file sama di kedua folder
assert len(files_folder_1) == len(files_folder_2), "Folder tidak memiliki jumlah file yang sama."

print("test before loop")
print(files_folder_1)
print(files_folder_2)

# Loop untuk perbandingan di sini, di luar loop konversi
for i in range(len(files_folder_1)):
    # print("test masuk loop")
    print(i)
    file_path_1 = os.path.join(folder_path_1, files_folder_1[i])
    file_path_2 = os.path.join(folder_path_2, files_folder_2[i])

    # print("test path")
    # Membaca gambar
    img1 = cv2.imread(file_path_1)
    img2 = cv2.imread(file_path_2)
    # print("test read")
    # Menghitung PSNR dan SSIM
    psnr_value = calculate_psnr(img1, img2)
    ssim_value = calculate_ssim(img1, img2)
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")

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

with open('meta_test_compare.json', 'w') as outfile:
    json.dump(json_data_with_meta, outfile)