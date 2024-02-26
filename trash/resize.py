import imagecodecs
import numpy as np
import cv2

ori = cv2.imread("a.jpeg")

print(np.shape(ori))

resized = cv2.resize(ori, (960, 540))

cv2.imwrite("a_s.jpeg", resized)