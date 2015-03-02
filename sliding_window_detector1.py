# Sliding window detector
# window size 320 x 243 (w x h)


import numpy as np
import cv2

from sklearn.preprocessing import normalize
from sklearn.externals import joblib
from skimage import color
from skimage.feature import hog
from PIL import Image
# from nolearn.dbn import DBN

model_clone = joblib.load('hog_dbn_models/hog_dbn_1.pkl')

path = 'test_images/1.jpg'
or_image = Image.open(path)
image = color.rgb2gray(np.asarray(or_image.resize((320,243))))


print(np.shape(image))

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