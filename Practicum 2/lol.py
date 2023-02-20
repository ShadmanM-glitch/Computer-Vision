import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("E:\\CSCI 4261\\Practicum 2\\sample.jpg")/255
img = img.astype(np.float32)
f, axarr = plt.subplots(1,3)
axarr[0].imshow(img)
axarr[1].imshow(img)
axarr[2].imshow(img)
plt.show()
