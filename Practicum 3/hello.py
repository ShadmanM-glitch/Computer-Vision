import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def fx(x, y):
        # Check bounds.
        x[ x < 0 ] = 0.
        y[ y < 0 ] = 0.

        x[ x > img.shape[1]-1 ] = img.shape[1]-1
        y[ y > img.shape[0]-1 ] = img.shape[0]-1

        return ggmix[ (y.round().astype(int), x.round().astype(int)) ]

def fy(x, y):
    # Check bounds.
    x[ x < 0 ] = 0.
    y[ y < 0 ] = 0.

    x[ x > img.shape[1]-1 ] = img.shape[1]-1
    y[ y > img.shape[0]-1 ] = img.shape[0]-1

    return ggmiy[ (y.round().astype(int), x.round().astype(int))]

def edge_gradient( img, sigma=30. ):
    # Gaussian smoothing.
    #smoothed = filt.gaussian_filter( (img-img.min()) / (img.max()-img.min()), sigma )
    # Gradient of the image in x and y directions.
    giy, gix = np.gradient( img )
    # Gradient magnitude of the image.
    gmi = (gix**2 + giy**2)**(0.5)
    # Normalize. This is crucial (empirical observation).
    gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())

    # Gradient of gradient magnitude of the image in x and y directions.
    ggmiy, ggmix = np.gradient( gmi )
    return ggmiy, ggmix


def iterate_snake(x, y, a, b, gamma=0.1, n_iters=10, return_all=True):
    N = x.shape[0]
    row = np.r_[-2*a - 6*b, a + 4*b,-b,np.zeros(N-5),-b,a + 4*b]
    A = np.zeros((N,N))
    for i in range(N):
        A[i] = np.roll(row, i)
    
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma*A)
    if return_all:
        snakes = []

    for i in range(n_iters):
        x_ = np.dot(B, x + gamma*fx(x,y))
        y_ = np.dot(B, y + gamma*fy(x,y))
        x, y = x_.copy(), y_.copy()
        if return_all:
            snakes.append( (x_.copy(),y_.copy()) )

    if return_all:
        return snakes
    else:
        return (x,y)

img = mpimg.imread('E:\\CSCI 4261\\Practicum 3\\image-1.png')
img = rgb2gray(img)

t = np.arange(0, 2*np.pi, 0.1)
x = 120+50*np.cos(t)
y = 140+60*np.sin(t)

alpha, beta, gamma, iterations = 0.001, 0.4, 100, 50

ggmiy, ggmix = edge_gradient(img)

snakes = iterate_snake(x,y,alpha,beta,gamma,iterations,True)

fig = plt.figure()
axarr  = fig.add_subplot(111)
axarr.imshow(img, cmap=plt.cm.gray)
axarr.set_xticks([])
axarr.set_yticks([])
axarr.set_xlim(0,img.shape[1])
axarr.set_ylim(img.shape[0],0)
axarr.plot(np.r_[x,x[0]], np.r_[y,y[0]], c=(0,0,0), lw=2)

for i, snake in enumerate(snakes):
    if i % 10 == 0:
        axarr.plot(np.r_[snake[0], snake[0][0]], np.r_[snake[1], snake[1][0]], c=(0,0,1), lw=2)


axarr.plot(np.r_[snakes[-1][0], snakes[-1][0][0]], np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1,0,0), lw=2)

plt.show()