import cv2
import pydicom
import os
import numpy as np
from pydicom.pixel_data_handlers.util import apply_color_lut, convert_color_space
from PIL import Image

im = cv2.imread('test_webp.webp')

ok = cv2.imwrite('test_jpeg.jpeg', im)

print(ok)