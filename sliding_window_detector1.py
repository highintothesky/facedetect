# Sliding window detector
# window size 320 x 243 (w x h)
# 50% overlap


import numpy as np
import cv2
import math

from sklearn.preprocessing import normalize
from sklearn.externals import joblib
from skimage import color
from skimage.feature import hog
from PIL import Image

# load model
model_clone = joblib.load('hog_dbn_models/hog_dbn_1.pkl')

# test image
path = 'test_images/1.jpg'
# or_image = Image.open(path)
# image = color.rgb2gray(np.asarray(or_image.resize((320,243))))
image = color.rgb2gray(np.asarray(Image.open(path)))


(target_height, target_width) = np.shape(image)

detect_height, detect_width = (243, 320)

overlap_height = int(math.floor(0.5*detect_height))
overlap_width = int(math.floor(0.5*detect_width))

print target_height, target_width
print overlap_height, overlap_width

# exit if target is too small to apply sliding windows
if(target_width < detect_width or target_height < detect_height):
	sys.exit()

# get the number of possible windows in target image
hor_windows = int(math.floor(target_width/overlap_width - 1))
ver_windows = int(math.floor(target_height/overlap_height - 1))
print hor_windows, ver_windows

for (x in hor_windows - 1):
	for (y in ver_windows - 1):
		im_window = image[y*overlap_height:detect_height+y*overlap_height , ]


# image_crop = image[0:243,0:320]
# image_res = image.resize((320,243))
# print(np.shape(image_crop))

fd = np.asarray(hog(image, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(3, 3), visualise=False))

fd_transpose = np.zeros((1,18954))
fd_transpose[:,:] = fd

print(fd.shape, fd_transpose.shape)

pred = model_clone.predict(normalize(fd_transpose))

print(pred)

cv2.imshow("crop", image)
cv2.waitKey(0)