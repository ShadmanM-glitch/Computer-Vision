import numpy as np
import matplotlib.pyplot as plt

def convolve(image: np.array, kernel: np.array) -> np.array:
    imageX, imageY, kernelX, kernelY = *image.shape, *kernel.shape
    resultX, resultY = imageX - kernelX + 1, imageY - kernelY + 1
    result = np.zeros((resultX, resultY), dtype = kernel.dtype)
    for i in range(resultX):
        for j in range(resultY):
            result[i, j] = (kernel * image[i : i + kernelX, j : j + kernelY]).sum()          
    return result


def gradCalculation(image):
    sobelX = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    sobelY = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    gradX = convolve(image, sobelX)
    gradY = convolve(image, sobelY)
    
    abs_grad_x=np.abs(gradX).astype('uint8')
    abs_grad_y=np.abs(gradY).astype('uint8')
    
    theta = np.arctan2(abs_grad_x, abs_grad_y)
    grad = (abs_grad_y * 0.5 + abs_grad_y * 0.5).astype("uint8")
    #grad_squared = grad**2
    return grad
    
def my_snake(image):
    
    origin1 = (149,214)
    
    
    
    return 0


image = plt.imread("E:\\CSCI 4261\\Practicum 3\\image-1.png")
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print(image.ndim)
output = gradCalculation(pixel_values)

plt.figure()
plt.imshow(output)
plt.show()

    




