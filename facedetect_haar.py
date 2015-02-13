import cv2

# cascade = "C:\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml"

threshold = 235
scaling_factor = 7
picture_taken = False
frame_name = "Face Tracking"

capture = cv2.VideoCapture(0)
cv2.namedWindow(frame_name)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


beer = cv2.imread('beer.jpg')

# if beer[0,0,0] > threshold:
#     print "ja"

for y_pix in range(0,beer.shape[0]):
    for x_pix in range(0,beer.shape[1]):
        # print y_pix, x_pix, beer[y_pix,x_pix,0]
        # if beer[y_pix,x_pix,0] > threshold :
        #     print "veel rood"
        if (beer[y_pix,x_pix,0] > threshold and beer[y_pix,x_pix,1] > threshold and beer[y_pix,x_pix,2] > threshold):
            beer[y_pix,x_pix,0] = 0
            beer[y_pix,x_pix,1] = 0
            beer[y_pix,x_pix,2] = 0



while cv2.waitKey(1) == -1:
    success, frame = capture.read()

    # print frame.shape
    # downsized_frame_template = (frame.shape[1]/scaling_factor, frame.shape[0] / scaling_factor)
    # downsized_frame = cv2.resize(frame, downsized_frame_template)
    # possible_faces = classifier.detectMultiScale(downsized_frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    possible_faces = classifier.detectMultiScale(gray, 1.35, 5)

    # print len(possible_faces)

    for (x,y,w,h) in possible_faces:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        beer_resized = cv2.resize(beer, (w,h))

        # print beer_resized.shape

        x_offset = y_offset = 50
        frame[y:y+h, x:x+w] = beer_resized
        # for c in range(0,3):
        #     frame[y:y+h, x:x+w] = beer_resized[:,:,c] * (beer_resized[:,:,3]/255.0)  + frame[y:y+h, x:x+w, c] * (1.0 - beer_resized[:,:,3]/255.0)


    cv2.imshow(frame_name, frame)