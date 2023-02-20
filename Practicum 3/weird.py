import morphsnakes

import numpy as np
from scipy.misc import imread
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as ppl

def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u

imgcolor = imread("E:\CSCI 4261\Practicum 3\image-1.png")/255.0
img = rgb2gray(imgcolor)

# MorphACWE does not need g(I)

# Morphological ACWE. Initialization of the level-set.
macwe = morphsnakes.MorphACWE(img, smoothing=3, lambda1=1, lambda2=1)
macwe1 = macwe
macwe.levelset = circle_levelset(img.shape, (218, 150), 25)
#macwe.levelset = circle_levelset(img.shape, (273, 439), 25)

# Visual evolution.
ppl.figure()
morphsnakes.evolve_visual(macwe, num_iters=250, background=imgcolor)

#ppl.figure(2)
#morphsnakes.evolve_visual(macwe1, num_iters=150, background=imgcolor)

