import numpy as np
import matplotlib.pyplot as plt
import cv2

T_r = np.array([[0.8660, 0.5, 0], [-0.5, 0.8660, 0], [0, 0, 1]])
T_s = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
T_tl = np.array([[1, 0, 50], [0, 1, 100], [0, 0, 1]])


def display(img,title):
    plt.figure(title)
    plt.imshow(img,cmap ="gray")
    plt.axis("off")

# TASK 1
def warping_affine(matrix,img,defaultx = 800,defaulty =600):
    img = np.reshape(img, img.shape + (1,))
    img_transformed = np.empty((defaultx,defaulty, 1), dtype=np.uint8)
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            pixel_data = img[i, j]
            input_coords = np.array([i, j, 1])
            i_out, j_out,_ = matrix @ input_coords
            img_transformed[int(i_out), int(j_out)] = pixel_data
    return img_transformed


# TASK 2 
    """following the theory of linear algebra matrix1 * matrix2 = matrix3
    therefore matrix1 * inverse matrix3 = matrix2
    """
def align_2dof(input,reference):
    inc = 50
    while(1):
        result = warping_affine(matrix,input)
        if(input == reference):
            break
        else:
            inc+=1
            matrix = np.array([[1, 0, inc], [0, 1, 156], [0, 0, 1]])  

    return matrix

# TASK 3
def affine_registration(input,reference):  
    
    height, width = reference.shape
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    keypoint1, d1 = sift.detectAndCompute(input,None)
    keypoint2, d2 = sift.detectAndCompute(reference,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(d1,d2,k=2)

    lowes_ratio_test = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            lowes_ratio_test.append(m)

    src_pts = np.float32([ keypoint1[m.queryIdx].pt for m in lowes_ratio_test ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoint2[m.trainIdx].pt for m in lowes_ratio_test ]).reshape(-1,1,2)
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
 
    transformed_img = cv2.warpPerspective(input,
                        homography, (width, height))
    
    if(np.array_equiv(transformed_img,input)): 
        pass
    else:
        affine_registration(transformed_img, reference)
    
    return homography, transformed_img
    
#Load Images
img1 = plt.imread('img1.jpg')
img2 = plt.imread('img2.jpg')
img3 = plt.imread('img3.jpg')

#Task 1 using affine matrix multiplication
translated_img = warping_affine(T_tl,img1)
rotated_img = warping_affine(T_r,img1)
scaled_img = warping_affine(T_s,img1)

#Task 2 muted since its very time consuming
#trial = align_2dof(img1,img2)

#Task 3 function being called
matrix, result = affine_registration(img1,img3)
print("The tranformation matrix that will give the resultant image is:")
print(matrix)

display(scaled_img,"scaled 50 percent")
display(rotated_img,"rotated 30 degrees")
display(translated_img,"translated image")
display(result,"img1 aligned task 3")
plt.show()

