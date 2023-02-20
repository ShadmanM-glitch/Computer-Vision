import numpy as np
import cv2
from matplotlib import pyplot as plt

def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # Read the video and first frame
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
dense_optical_flow()
cap = cv2.VideoCapture("traffic.mp4")
backgroundobject = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = np.ones((3,3),np.uint8)

lower_val = (10, 0, 0)
upper_val = (135,255,255)

low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

color = np.random.randint(0, 255, (100, 3))

variance_list = [300,150]



while True:
    # Read the next frame
    ret, new_frame = cap.read()
    frame_copy = new_frame
    if not ret:
        break

    # Preprocessing for exact method
    
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    flow = method(old_frame, new_frame, None, *params)

    # Encoding: convert the algorithm's output into Polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Use Hue and Value to encode the Optical Flow
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV image into BGR for demo
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("frame", frame_copy)
    cv2.imshow("optical flow", bgr)
    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        break

    # Update the previous frame
    old_frame = new_frame

cap.release()
cv2.destroyAllWindows()