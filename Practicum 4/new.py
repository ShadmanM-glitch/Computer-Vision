import numpy as np
import matplotlib.pyplot as plt

T_r = np.array([[0.8660, 0.5, 0], [-0.5, 0.8660, 0], [0, 0, 1]])
T_s = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
T_tl = np.array([[1, 0, 400], [0, 1, 100], [0, 0, 1]])
T_i = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
matrix = np.array([[1, 0, 50], [0, 1, -50], [0, 0, 1]])

def warping_affine(matrix_to_apply, image_map):
    #takes an image and matrices and applies it.  
    x_min = 0
    y_min = 0
    x_max = image_map.shape[0]
    y_max = image_map.shape[1] 

    new_image_map = np.zeros((x_max, y_max), dtype=int)

    for y_counter in range(0, y_max):
        for x_counter in range(0, x_max):
            curr_pixel = [x_counter,y_counter,1]

            curr_pixel = np.dot(matrix_to_apply, curr_pixel)

            # print(curr_pixel)

            if curr_pixel[0] > x_max - 1 or curr_pixel[1] > y_max - 1 or x_min > curr_pixel[0] or y_min > curr_pixel[1]:
                next
            else:
                new_image_map[x_counter][y_counter] = image_map[int(curr_pixel[0])][int(curr_pixel[1])] 

    return new_image_map


img1 = plt.imread('E:\\CSCI 4261\\Practicum 4\\img1.jpg')
img2 = plt.imread('E:\\CSCI 4261\\Practicum 4\\img2.jpg')
identity = warping_affine(T_i, img2)

translated_img = warping_affine(T_tl,img1)
rotated_img = warping_affine(T_r,img1)
scaled_img = warping_affine(T_s,img1)
result = warping_affine(matrix,img1)


plt.figure("img2")
plt.imshow(identity,cmap ="gray")
plt.figure("img1 translated")
plt.imshow(translated_img,cmap ="gray")
plt.show()