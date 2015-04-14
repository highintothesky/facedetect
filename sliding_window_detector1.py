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
path = 'test_images/7.gif'
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
	print("too small!")
	sys.exit()

# get the number of possible windows in target image
hor_windows = int(math.floor(target_width/overlap_width - 1))
ver_windows = int(math.floor(target_height/overlap_height - 1))
print hor_windows, ver_windows

for x in range(0, hor_windows):
	for y in range(0, ver_windows):
		im_window = image[y*overlap_height:detect_height+y*overlap_height , x*overlap_width:detect_width + x*overlap_width]
		fd = np.asarray(hog(im_window, orientations=9, pixels_per_cell=(16, 16),
		                    cells_per_block=(3, 3), visualise=False))

		fd_transpose = np.zeros((1,18954))
		fd_transpose[:,:] = fd

		pred = model_clone.predict(normalize(fd_transpose))
		print pred
		if(pred == [1]):
			cv2.imshow("hit!", im_window)
			cv2.waitKey(0)


# image_crop = image[0:243,0:320]
# image_res = image.resize((320,243))
# print(np.shape(image_crop))

# fd = np.asarray(hog(im_window, orientations=9, pixels_per_cell=(16, 16),
#                     cells_per_block=(3, 3), visualise=False))

# fd_transpose = np.zeros((1,18954))
# fd_transpose[:,:] = fd

# print(fd.shape, fd_transpose.shape)

# pred = model_clone.predict(normalize(fd_transpose))

# print(pred)

# cv2.imshow("crop", image)
# cv2.waitKey(0)