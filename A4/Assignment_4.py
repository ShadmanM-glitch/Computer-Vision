import numpy as np
import cv2

cap = cv2.VideoCapture("traffic.mp4")
output_file = "E:\\CSCI 4261\\A4\\track.mp4"
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50,50),
                   maxLevel = 2,
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

is_begin = True

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret,frame = cap.read()
    processed = frame
    roi = frame[250:620, 100:350]
    
    fgmask = fgbg.apply(roi)
    
    contours, _ = cv2.findContours(fgmask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        #cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
        area = cv2.contourArea(cnt)
        if area > 2500: #and area < 1900: 
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w,y+h),(0,255,0),3)
    
    if is_begin:
        h, w, _ = processed.shape
        out = cv2.VideoWriter(output_file, fourcc, 30, (w, h), True)
        is_begin = False

    
    
    out.write(frame)
        
    cv2.imshow('frame',frame)
    cv2.imshow('roi',roi)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    


cv2.destroyAllWindows()
cap.release()