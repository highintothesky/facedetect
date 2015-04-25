import numpy as np
import cv2
import time

from sklearn.preprocessing import normalize
from skimage.feature import hog
# from cv2 import HOGDescriptor #, normalize
from sklearn.externals import joblib
from skimage import color
# from sklearn.preprocessing import normalize

# 640, 360

def hog3(input_image):
    return hog(input_image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(3, 3), visualise=False)

# load model
model_clone = joblib.load('hog_dbn_models/hog_dbn_1.pkl')


cap = cv2.VideoCapture(0)
cap.set(3,384)
cap.set(4,286)

# hog = HOGDescriptor((8,8), (8,8), (8,8), (8,8), 9)

# fps and detection counters
frame_counter = 0
timer = time.time()
detected = 0

while(True):
    # display fps
    new_timer = time.time()
    if new_timer > timer + 10:
        print "frames per second: ", frame_counter/10.0
        frame_counter = 0
        timer = new_timer
    frame_counter += 1


    # Capture frame-by-frame
    ret, frame = cap.read()

    # print frame[0,0]

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = color.rgb2gray(np.asarray(frame))

    # print gray.shape, type(gray)

    # compute hog
    # h1 = hog.compute(normalize(gray).astype('uint8'))
    # h1 = hog.compute(gray.astype('uint8'))
    h1 = hog3(gray)
    # print gray.shape, h1.shape
    # print np.mean(h1), max(h1)

    # fd_transpose = np.zeros((1,hog_length))
    # fd_transpose[:,:] = h1.T

    # print fd_transpose.shape

    pred = model_clone.predict(normalize(h1.T))
    # pred = model_clone.predict(h1.T)

    if(pred == [1]):
        print "detection! ", pred, detected
        detected += 1

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()