# facedetect - HOG + dbn

from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from skimage import color
from skimage.feature import hog
from cv2 import HOGDescriptor #, normalize
from nolearn.dbn import DBN
from os import listdir
from PIL import Image
import numpy as np
import cv2

# opencv hog is much faster!
# yet yields shitty results...

# good image database:
# http://groups.csail.mit.edu/vision/SUN/

# hog-implementation in python:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
# probably quicker:
# http://codereview.stackexchange.com/questions/42763/histogram-of-oriented-gradients-hog-feature-detector-for-computer-vision-tr


# fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualise=True)



# create the hog descriptor
# hog2 = HOGDescriptor((8,8), (8,8), (8,8), (8,8), 9)
def hog3(input_image):
	return hog(input_image, orientations=9, pixels_per_cell=(16, 16),
		cells_per_block=(3, 3), visualise=False)


# make a list of all images in the dir
images = listdir("/Users/nibo/data/diy_face")
images.pop(0)

hog_length = 26730
height, width = (286, 384)

# create feature vector matrix
data_mat = np.zeros((len(images), hog_length))
positive_Y = np.ones(len(images),dtype=np.int)
current_column = 0

print("positive data matrix size:", data_mat.shape)

for im in images:
	path = "/Users/nibo/data/diy_face/" + im
	original_image = Image.open(path)

	# pix = original_image.load()
	# print pix[0,0]

	if original_image.size != (width, height):
		image = color.rgb2gray(np.asarray(original_image.resize((width,height))))
	else:
		image = color.rgb2gray(np.asarray(original_image))
	# fd = hog(image, orientations=9, pixels_per_cell=(16, 16),
	#                     cells_per_block=(3, 3), visualise=False)
	# fd = hog2.compute(image.astype('uint8'))
	fd = hog3(image)
	# print im, np.mean(fd), max(fd)
	data_mat[current_column,:] = fd.T
	current_column += 1

print("processed " + str(current_column) + " images for the positive data set")


# make a list of all images in the negative dir
neg_images = listdir("/Users/nibo/data/diy_background")
neg_images.pop(0)
neg_images.pop(-1)


# create feature vector matrix for negative samples
neg_data_mat = np.zeros((len(neg_images), hog_length))
negative_Y = np.zeros(len(neg_images), dtype=np.int)
current_column = 0

print("negative data matrix size:", neg_data_mat.shape)

for neg_im in neg_images:
	path = "/Users/nibo/data/diy_background/" + neg_im
	original_image = Image.open(path)
	if original_image.size != (width, height):
		image = color.rgb2gray(np.asarray(original_image.resize((width,height))))
	else:
		image = color.rgb2gray(np.asarray(original_image))
	# fd = hog(neg_image_resized, orientations=9, pixels_per_cell=(16, 16),
	#                     cells_per_block=(3, 3), visualise=False)
	# fd = hog2.compute(image.astype('uint8'))
	fd = hog3(image)
	neg_data_mat[current_column,:] = fd.T
	current_column += 1

print("processed " + str(current_column) + " images for the negative data set")

total_X_prenorm = np.concatenate([data_mat, neg_data_mat])
total_Y = np.concatenate([positive_Y, negative_Y])

# total_X = normalize(total_X_prenorm)
total_X = normalize(total_X_prenorm)

(trainX, testX, trainY, testY) = train_test_split(total_X, total_Y, test_size=0.33, random_state=42)
print(trainX.shape)


# train the Deep Belief Network with hog_length input units, 
# 1000 hidden units, 2 output units (one for
# each possible output classification)
dbn = DBN(
	[trainX.shape[1], 1000, 2],
	learn_rates = 0.3,
	learn_rate_decays = 0.7,
	epochs = 5,
	verbose = 1)

# train multicore
dbn.fit(trainX, trainY)

# compute the predictions for the test data and show a classification
# report
preds = dbn.predict(testX)
print(classification_report(testY, preds))





#### TEST ####

test_path = 'test_images/7.gif'
original_test_image = Image.open(test_path)
test_image_resized = color.rgb2gray(np.asarray(original_test_image.resize((width,height))))


# print neg_image_resized.shape, type(neg_image_resized)
# print test_image_resized.shape, type(test_image_resized)

# test_fd = hog2.compute(test_image_resized.astype('uint8'))
test_fd = hog3(test_image_resized)

# fd_transpose = np.zeros((1,hog_length))
# fd_transpose[:,:] = test_fd.T

# pred_test = dbn.predict(normalize(fd_transpose))
pred_test = dbn.predict(normalize(test_fd.T))


not_test_path = 'test_images/not1.jpg'
original_not_test_image = Image.open(not_test_path)
not_test_image_resized = color.rgb2gray(np.asarray(original_not_test_image.resize((width,height))))

# not_test_fd = hog2.compute(not_test_image_resized.astype('uint8'))
not_test_fd = hog3(not_test_image_resized)

# not_fd_transpose = np.zeros((1,hog_length))
# not_fd_transpose[:,:] = not_test_fd.T

# not_pred_test = dbn.predict(normalize(not_fd_transpose))
not_pred_test = dbn.predict(normalize(not_test_fd.T))


print("prediction test: ", pred_test, "negative test: ", not_pred_test)
print("shape test data: ", np.shape(testX), "shape neg. test: ", np.shape(not_test_fd.T))
# print("prediction test: ", pred_test, "negative test: ", not_pred_test)
# print("shape test data: ", np.shape(testX), "shape neg. test: ", np.shape(not_fd_transpose))
# print("mean test data: ", np.mean(testX, axis = 1), "mean pred. test: ", np.mean(fd_transpose, axis = 1), "mean neg. pred. test: ", np.mean(not_fd_transpose, axis = 1))


#### WRITE MODEL ####


joblib.dump(dbn, 'hog_dbn_models/hog_dbn_1.pkl', compress=3)

# # cv2.imshow("sub03", hog_image)
# cv2.waitKey(0)