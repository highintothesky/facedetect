# facedetect - HOG + dbn

from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from skimage import color
from skimage.feature import hog
from nolearn.dbn import DBN
from os import listdir
from PIL import Image
import numpy as np
import cv2



# fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualise=True)



# make a list of all images in the dir
images = listdir("/Users/nibo/data/yalefaces")
images.pop(0)


# create feature vector matrix
data_mat = np.zeros((len(images), 18954))
positive_Y = np.ones(len(images),dtype=np.int)
current_column = 0

print("positive data matrix size:", data_mat.shape)

for im in images:
	path = "/Users/nibo/data/yalefaces/" + im
	image = color.rgb2gray(np.asarray(Image.open(path)))
	fd = hog(image, orientations=9, pixels_per_cell=(16, 16),
	                    cells_per_block=(3, 3), visualise=False)
	data_mat[current_column,:] = fd
	current_column += 1

print("processed " + str(current_column) + " images for the positive data set")


# make a list of all images in the negative dir
neg_images = listdir("/Users/nibo/data/Caltech/PNGImages/background")
neg_images.pop(0)
neg_images.pop(-1)


# create feature vector matrix for negative samples
neg_data_mat = np.zeros((len(neg_images), 18954))
negative_Y = np.zeros(len(neg_images), dtype=np.int)
current_column = 0

print("negative data matrix size:", neg_data_mat.shape)

for neg_im in neg_images:
	path = "/Users/nibo/data/Caltech/PNGImages/background/" + neg_im
	# first open
	original_neg_image = Image.open(path)
	# then resize
	neg_image_resized = color.rgb2gray(np.asarray(original_neg_image.resize((320,243))))
	fd = hog(neg_image_resized, orientations=9, pixels_per_cell=(16, 16),
	                    cells_per_block=(3, 3), visualise=False)
	neg_data_mat[current_column,:] = fd
	current_column += 1

print("processed " + str(current_column) + " images for the negative data set")

total_X_prenorm = np.concatenate([data_mat, neg_data_mat])
total_Y = np.concatenate([positive_Y, negative_Y])

total_X = normalize(total_X_prenorm)


(trainX, testX, trainY, testY) = train_test_split(total_X, total_Y, test_size=0.33, random_state=42)
print(trainX.shape)

# train the Deep Belief Network with 18954 input units, 
# 1000 hidden units, 2 output units (one for
# each possible output classification)
dbn = DBN(
	[trainX.shape[1], 1000, 2],
	learn_rates = 0.3,
	learn_rate_decays = 0.9,
	epochs = 3,
	verbose = 1)
dbn.fit(trainX, trainY)

# compute the predictions for the test data and show a classification
# report
preds = dbn.predict(testX)
print(classification_report(testY, preds))



#### TEST ####

test_path = 'test_images/7.gif'
original_test_image = Image.open(test_path)
test_image_resized = color.rgb2gray(np.asarray(original_test_image.resize((320,243))))
# image_crop = test_image_resized[0:243,0:320]

test_fd = np.asarray(hog(test_image_resized, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(3, 3), visualise=False))

fd_transpose = np.zeros((1,18954))
fd_transpose[:,:] = test_fd

pred_test = dbn.predict(normalize(fd_transpose))



not_test_path = 'test_images/not1.jpg'
original_not_test_image = Image.open(not_test_path)
not_test_image_resized = color.rgb2gray(np.asarray(original_not_test_image.resize((320,243))))
# not_image_crop = not_test_image[0:243,0:320]

not_test_fd = np.asarray(hog(not_test_image_resized, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(3, 3), visualise=False))

not_fd_transpose = np.zeros((1,18954))
not_fd_transpose[:,:] = not_test_fd

not_pred_test = dbn.predict(normalize(not_fd_transpose))

print("prediction test: ", pred_test, "negative test: ", not_pred_test)
print("shape test data: ", np.shape(testX), "shape neg. test: ", np.shape(not_fd_transpose))
# print("mean test data: ", np.mean(testX, axis = 1), "mean pred. test: ", np.mean(fd_transpose, axis = 1), "mean neg. pred. test: ", np.mean(not_fd_transpose, axis = 1))



# joblib.dump(dbn, 'hog_dbn_models/hog_dbn_1.pkl', compress=9)

# cv2.imshow("sub03", hog_image)
# cv2.waitKey(0)