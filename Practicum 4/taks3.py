import cv2
import matplotlib.pyplot as plt
import numpy as np

# Open the image files.
img1 = plt.imread("img1.jpg") # Image to be aligned.
img2 = plt.imread("img2.jpg") # Reference image.


height, width = img2.shape

sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, d1 = sift.detectAndCompute(img1,None)
kp2, d2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)
# Match features between the two images.
# We create a Brute Force matcher with
# Hamming distance as measurement mode.
matches = flann.knnMatch(d1,d2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
transformed_img = cv2.warpPerspective(img1,
                    homography, (width, height))
# Save the output.
print(homography)
plt.figure()
plt.imshow(transformed_img, cmap= "gray")
plt.show()
