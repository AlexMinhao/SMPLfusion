import cv2

cap = cv2.VideoCapture('D:\\Research\\totalcapture\\mattes\\S1\\walking1\\TC_S1_walking1_cam1.mp4')
success, image = cap.read()
count = 0
success = True
while success:
    # save frame as JPEG file
    cv2.imwrite("D:\\Research\\totalcapture\\frames\\frame%d.jpg" % count, image)
    success, image = cap.read()
    print('Read a new frame: ', count)
    count += 1