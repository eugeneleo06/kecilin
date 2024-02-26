import imagecodecs
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# def psnr(img1, img2):
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     max_val = 255.0
#     return 20 * np.log10(max_val / np.sqrt(mse))


img_ori = imagecodecs.imread('mri_single_jpeg/mri_0.jpeg')
img_comp = imagecodecs.imread('mri_single/mri_0.webp')

img_comp_gray = cv2.cvtColor(img_comp, cv2.COLOR_BGR2GRAY)

print(np.shape(img_ori))
print(np.shape(img_comp_gray))


img_ori_arr = np.asarray(img_ori, dtype=np.float32)
img_comp_arr = np.asarray(img_comp_gray, dtype=np.float32)

psnr_value = psnr(img_ori_arr, img_comp_arr, data_range=255)

# Calculate SSIM
ssim_value, _ = ssim(img_ori_arr, img_comp_arr, full=True, data_range=255)

print("PSNR:", psnr_value)
print("SSIM:", ssim_value)
