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

CONST_JPEG = '.JPEG'
CONST_WEBP = '.WEBP'
CONST_JXL = '.JXL'

quality = [100,90,80,70,60,50,40]
ext = ['.JPEG', '.WEBP', '.JXL']

# Path to your single DICOM file
file_path = '/home/baptista/Documents/kecilin/automation3/mri3.dcm'

# Read the DICOM file
dicom = pydicom.dcmread(file_path)

pixel = dicom.pixel_array

bmp_imgs = []
for i in range(pixel.shape[0]):
    img = pixel[i].astype(float)
    
    rscl_img = (np.maximum(img, 0) / img.max()) * 255
    final_img = np.uint8(rscl_img)
    
    # im = Image.fromarray(final_img)
    # im.save("automation/bmp/mri_" + str(i+1) + ".bmp", 'BMP')
    bmp_filename = 'automation3/bmp/mri_' + str(i+1) + '.bmp'
    cv2.imwrite(bmp_filename, final_img)
    print(bmp_filename)
    for x in ext:
        for y in quality:
            filename = "automation3/" + x[1:].lower() + "_" + str(y) + "/mri_" + str(i+1) + x.lower()
            # q = cv2.IMWRITE_JPEG_QUALITY
            if x == CONST_WEBP:
                q = cv2.IMWRITE_WEBP_QUALITY
                cv2.imwrite(filename, final_img, [int(q), y])
            elif x == CONST_JPEG:
                r = os.system('cjpeg -quality ' + str(y) + ' -outfile ' + filename + ' ' + bmp_filename)
            elif x == CONST_JXL:
                im = Image.open(bmp_filename)
                im.save('temp.png')
                r = os.system('./cjxl -q '+ str(y) + ' temp.png ' + filename + ' --quiet')
                os.remove('temp.png')
            if r != 0:
                quit()

def get_paths(name: str, format: str = None):
    # name = jpeg_90
    # format = .jpeg
    if name == "bmp":
        files = glob.glob('automation3/'+ name + '/*.bmp')
    else:
        files = glob.glob('automation3/'+ name + '/*' + format.lower())
    n = len(files)
    paths = []
    for i in range (n):
        if name == "bmp":
            paths.append("automation3/" + name + "/mri_" + str(i+1) + ".bmp")
        else:
            paths.append("automation3/" + name + "/mri_" + str(i+1) + format.lower())
    return paths

def compare_psnr_ssim(bmp_files, files, ext=None):
    psnr_arr = []
    ssim_arr = []
    size_arr = []

    n = len(bmp_files)

    for x in range (n):
        img_ori = cv2.imread(bmp_files[x])
        if ext == CONST_JXL:
            img_comp = imagecodecs.imread(files[x])
        else:
            img_comp = cv2.imread(files[x])
        
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        if img_comp.shape != (512,512):
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

def generate_group(key):
    if key.startswith('jpeg'):
        group = 'jpeg'
    elif key.startswith('webp'):
        group = 'webp'
    elif key.startswith('jxl'):
        group = 'jxl'
    return group

data, df, seq = {}, {}, []
files, psnrData, ssimData, sizeData = {}, {}, {}, {}
for i in range(pixel.shape[0]):
    seq.append(i+1)

files["bmp"] = get_paths('bmp')

for x in ext:
    for y in quality:
        files[generate_key(x,y)] = get_paths(generate_key(x,y), x)
        psnrData[generate_key(x,y)], ssimData[generate_key(x,y)], sizeData[generate_key(x,y)] = compare_psnr_ssim(files['bmp'], files[generate_key(x,y)], x)
        data[generate_key(x,y)] = {
            'Sequence': seq,
            'PSNR': psnrData[generate_key(x,y)],
            'SSIM': ssimData[generate_key(x,y)],
            'Size (KB)': sizeData[generate_key(x,y)],
        }
        df[generate_key(x,y)] = pd.DataFrame(data[generate_key(x,y)])

# Specify the Excel file path
excel_file_path = 'automation3/result.xlsx'

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
plt.savefig('automation3/plot/psnr.png')

plt.figure(figsize=(12, 6))
for x in ext:
    for y in quality:
        plt.plot(seq, df[generate_key(x,y)]['SSIM'], label=generate_key(x,y), linestyle='-')
plt.title('SSIM Comparison')
plt.xlabel('Sequence')
plt.ylabel('SSIM')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('automation3/plot/ssim.png')

plt.figure(figsize=(12, 6))
for x in ext:
    for y in quality:
        plt.plot(seq, df[generate_key(x,y)]['Size (KB)'], label=generate_key(x,y), linestyle='-')
plt.title('Size Comparison')
plt.xlabel('Sequence')
plt.ylabel('Size (KB)')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('automation3/plot/size.png')

avg = {}

for x in ext:
    for y in quality:
        data = {}
        psnr_avg = df[generate_key(x,y)]['PSNR']
        ssim_avg = df[generate_key(x,y)]['SSIM']
        size_avg = df[generate_key(x,y)]['Size (KB)']
        data['PSNR'] = sum(psnr_avg) / len(psnr_avg)
        data['SSIM'] = sum(ssim_avg) / len(ssim_avg)
        data['Size (KB)'] = sum(size_avg) / len(size_avg)
        avg[generate_key(x,y)] = data

colors = {'jpeg': 'blue', 'webp': 'red', 'jxl':'green'}  

# ================== PSNR =================
group_coords = {}
for x in ext:
    group_coords[x[1:].lower()] = {'x': [], 'y': []}
labels = []

for x in ext:
    for y in quality:
        key = generate_key(x, y)
        group = generate_group(key)
        group_coords[group]['x'].append(avg[key]['Size (KB)'])
        group_coords[group]['y'].append(avg[key]['PSNR'])
        labels.append(key)
        
plt.figure(figsize=(12, 6))

# Plot lines for each group first
for group, coords in group_coords.items():
    # Plot line connecting points in this group with a generic label for the group
    plt.plot(coords['x'], coords['y'], color=colors[group], zorder=1)
    # line_label = True  # Ensure the group line label is added only once to the legend

# Then plot each point individually for the unique legends
for x in ext:
    for y in quality:
        key = generate_key(x, y)
        group = generate_group(key)
        x_coord = avg[key]['Size (KB)']
        y_coord = avg[key]['PSNR']
        color = colors[group]

        # Plot the point with a unique label
        plt.scatter(x_coord, y_coord, color=color, s=20, zorder=2, label=key)

# Handling the legend
# Extract handles and labels and keep unique labels only
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = []
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)

plt.title('PSNR Comparison')
plt.xlabel('Size (KB)')
plt.ylabel('PSNR')
plt.legend(unique_handles, unique_labels, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(True)
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.tight_layout()
plt.savefig('automation3/plot/psnr_size.png')


 # ================== SSIM =================
group_coords = {}
for x in ext:
    group_coords[x[1:].lower()] = {'x': [], 'y': []}

labels = []
for x in ext:
    for y in quality:
        key = generate_key(x, y)
        group = generate_group(key)
        group_coords[group]['x'].append(avg[key]['Size (KB)'])
        group_coords[group]['y'].append(avg[key]['SSIM'])
        labels.append(key)
        
plt.figure(figsize=(12, 6))
# Plot lines for each group first
for group, coords in group_coords.items():
    # Plot line connecting points in this group with a generic label for the group
    plt.plot(coords['x'], coords['y'], color=colors[group], zorder=1)

# Then plot each point individually for the unique legends
for x in ext:
    for y in quality:
        key = generate_key(x, y)
        group = generate_group(key)
        x_coord = avg[key]['Size (KB)']
        y_coord = avg[key]['SSIM']
        color = colors[group]

        # Plot the point with a unique label
        plt.scatter(x_coord, y_coord, color=color, s=20, zorder=2, label=key)

# Handling the legend
# Extract handles and labels and keep unique labels only
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = []
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)

plt.title('SSIM Comparison')
plt.xlabel('Size (KB)')
plt.ylabel('SSIM')
plt.legend(unique_handles, unique_labels, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(True)
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.tight_layout()  
plt.savefig('automation3/plot/ssim_size.png')