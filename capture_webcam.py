import numpy as np
import cv2
import time

# from cv2 import HOGDescriptor, normalize
from skimage import color
# from sklearn.preprocessing import normalize

# 640, 360


cap = cv2.VideoCapture(0)
cap.set(3,384)
cap.set(4,286)

capturecounter = 122

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = color.rgb2gray(np.asarray(frame))


    # print gray.shape, h1.shape


    # Display the resulting frame
    cv2.imshow('frame',gray)

    # capture frame
    if cv2.waitKey(1) & 0xFF == ord('c'):
        name = 'new_data/diy_img-' + str(capturecounter) + '.png'
        cv2.imwrite(name, frame)
        print "captured! ", capturecounter
        capturecounter += 1
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()