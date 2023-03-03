# -*- coding: utf-8 -*-
"""
    File name: imageFT_1.py
    Author: Maria Paula Rey, EAFIT University
    Email: mpreyb@eafit.edu.co
    Date last modified: 22/03/2022
    Python Version: 3.8
"""


import numpy as np
from scipy import misc
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2 

#------------------------------------------------------------------------------------------------------------
# Blurring (Gaussian)

img = cv2.imread('LenaColor.jpg',0)   # Loading image
imgr = cv2.resize(img, (400, 400))    # resizing


gauss = cv2.GaussianBlur(imgr,(3, 3),   20,    sigmaY = 1) #std in each dir  
# 20 is standar deviation. (5,5) tamaño de la ventana (kernel)
##Gaussian blur

#cv2.imshow('Original',imgr)          # Window for the original image
cv2.imshow('imgr',imgr)
#cv2.imshow('FILTRO GAUUSIANO',gauss) # Window for the proceesed image
cv2.imshow('gauss', gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey()

#------------------------------------------------------------------------------------------------------------


# =============================================================================
# The Discrete Fourier transform (DFT) and, by extension, 
# the FFT (which computes the DFT) have the origin in the first element (for an image, the top-left pixel) 
# for both the input and the output. 
# This is the reason we often use the fftshift function on the output, 
# so as to shift the origin to a location more familiar to us (the middle of the image).
# This means that we need to transform a 3x3 uniform weighted blurring kernel to look like this 
# before passing it to the FFT function:
    # 1/9  1/9  0  0  ... 0  1/9
    # 1/9  1/9  0  0  ... 0  1/9
    #  0    0  0  0  ... 0    0
    # ...  ...               ...
    #  0    0  0  0  ... 0    0
    # 1/9  1/9  0  0  ... 0  1/9
# =============================================================================

#img = misc.face()[:,:,0]               # Loading image.
img2 = cv2.imread('LenaColor.jpg',0)     # We could also process this image
img = cv2.resize(img2, (400, 400))       # resizing



kernel = np.ones((3,3)) / 9   

# =============================================================================
# KERNEL PADDING
# When padding the kernel, we need to take care that the origin (middle of the kernel) 
# is at location k_im.shape // 2 (integer division), within the kernel image k_im. 
# Initially the origin is at [3,3]//2 == [1,1]. 
# The image whose size we're matching is even in size,thus
# the origin there will be at [400,400]//2 == [200,200]. 
# This means that we need to pad a different amount to the left and to the right (and bottom and top). 
# We need to be careful computing this padding:
# =============================================================================


sz = (img.shape[0] - kernel.shape[0], img.shape[1] - kernel.shape[1])      # total amount of padding

#Padding the kernel
kernel = np.pad(kernel, (((sz[0]+1)//2, sz[0]//2), ((sz[1]+1)//2, sz[1]//2)), 'constant')

# Kernel shift
kernel = fftpack.ifftshift(kernel) 
                              
# Multiply kernel transform and image transform. then take the inverse transform of this result.
filtered = np.real(fftpack.ifft2(fftpack.fft2(img) * fftpack.fft2(kernel)))


# Comparing input image and output image
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(filtered, cmap = 'gray')
plt.title('Output image'), plt.xticks([]), plt.yticks([])
plt.show()

#------------------------------------------------------------------------------------------------------------
# Comparing resutls
plt.subplot(121),plt.imshow(gauss, cmap = 'gray')
plt.title('Gaussiang Blurring'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(filtered, cmap = 'gray')
plt.title('FT Method'), plt.xticks([]), plt.yticks([])
plt.show()
