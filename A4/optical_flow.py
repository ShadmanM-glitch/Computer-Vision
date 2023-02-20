import numpy as np
import cv2
from matplotlib import pyplot as plt


cap = cv2.VideoCapture("traffic.mp4")
backgroundobject = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = np.ones((3,3),np.uint8)
output_file = "E:\\CSCI 4261\\A4\\flow.mp4"
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

lower_val = (10, 0, 0)
upper_val = (135,255,255)


color = np.random.randint(0, 255, (100, 3))

variance_list = [300,150]

lk_params = dict(winSize  = (15, 15),
		maxLevel = 2,
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 100,
			qualityLevel = 0.3,
			minDistance = 7,
			blockSize = 7)


trajectory_len = 1000
detect_interval = 10
trajectories = []
frame_idx = 0
is_begin = True
while True:
	suc, frame = cap.read()
	resized_frame = cv2.resize(frame, None, fx=1, fy=1, interpolation = cv2.INTER_LINEAR)# resize the frame 
	org_frame = resized_frame.copy()    
	resized_frame = cv2.GaussianBlur(resized_frame, (7,7), 0)#blur
	processed = frame
	imgHSV = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)#tohsv

	sub = backgroundobject.apply(frame) #background subtraction
	contours , hi = cv2.findContours(sub, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)# finding contours
	if is_begin:
		h, w, _ = processed.shape
		out = cv2.VideoWriter(output_file, fourcc, 30, (w, h), True)
		is_begin = False
		

	# Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
	if len(trajectories) > 0:
		img0, img1 = prev_gray, sub
		p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
		p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
		p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
		d = abs(p0-p0r).reshape(-1, 2).max(-1)
		good = d < 1

		new_trajectories = []

		# Get all the trajectories
		for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
			if not good_flag:
				continue
			trajectory.append((x, y))
			if len(trajectory) > trajectory_len:
				del trajectory[0]
			new_trajectories.append(trajectory)
			# Newest detected point
			cv2.circle(org_frame, (int(x), int(y)), 2, (0, 0, 255), -1)

		trajectories = new_trajectories
		
		# Draw all the trajectories
		#cv2.polylines(org_frame, [np.int32(trajectory) for trajectory in trajectories], False, (0,255,0), 0)
		#cv2.putText(org_frame, 'track count points: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


	# Update interval - When to update and detect new features
	if frame_idx % detect_interval == 0:
		mask = np.zeros_like(sub)
		mask[:] = 255

		# Lastest point in latest trajectory
		for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
			#cv2.circle(mask, (x, y), 5, 0, -1)
			cv2.arrowedLine(mask, (x, y), (x+y,y+2), 0, 1)

		# Detect the good features to track
		p = cv2.goodFeaturesToTrack(sub, mask = mask, **feature_params)
		if p is not None:
			# If good features can be tracked - add that to the trajectories
			for x, y in np.float32(p).reshape(-1, 2):
				trajectories.append([(x, y)])

	

	frame_idx += 1
	prev_gray = sub
	out.write(org_frame)
	# Show Results
	cv2.imshow('Optical Flow', org_frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()