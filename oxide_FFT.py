import os
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
from scipy import fftpack
import re
import csv
import sys
import pandas as pd
import pylab as py
from skimage.filters import window
from scipy import signal
from scipy.ndimage.filters import uniform_filter1d
from skimage.filters import threshold_otsu


file_name = '2k_2.25_1x4_hs.png'  # Take a file name from the list
# 2b_2.25_5x5_hs.png
# 2i_2.25_3x3_hs.png
# 2k_2.25_1x4_hs.png

image_size = 8  # provide the image size in microns
crop_mask = 0  # crop if 0, mask if 1
segment = 0  # 0 to FFT the grayscale image, 1 to binarise the image first
comparison = 1  # makes a final 2 figures of both two rectangles and their respective FFTs, assumes c_m = 0

rectangle_x_length = 180
rectangle_y_length = 300
rectangle_x_origin = 250
rectangle_y_origin = 0
#2k use 180, 300, 250, 0

if comparison == 1:  # enter the coordinates of a second rectangle you'll use for masking
    rectangle2_x_length = 220
    rectangle2_y_length = 390
    rectangle2_x_origin = 230
    rectangle2_y_origin = 0

image_loc = 'Oxide Data/Test Set/' + file_name

def rgb2gray(rgb):  # Define a function that converts PNGs to grayscale arrays
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
def normalise(array):  # Set an image numpy array's values to be between 0 and 1
    norm_array = (array-np.min(array))\
                         / (np.max(array)-np.min(array))
    return norm_array
def radial_profile2(data, center=None, binning=2):
    y, x = np.indices((data.shape))  # first determine radii of all pixels

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # radius of the image.
    r_max = np.max(r)
    bin_no = np.rint(r_max/binning).astype(int)

    ring_brightness, radius = np.histogram(r, weights=data, bins=bin_no)
    # plt.plot(radius[1:], ring_brightness)
    # plt.show()
    return ring_brightness
def crop(image,x_origin,y_origin,x_length,y_length):
    x, y = image.shape
    zero_array = np.zeros([x, y])
    zero_array[x_origin:x_origin+x_length+1, y_origin:y_origin+y_length+1] = 1
    cropped_array = zero_array * image
    return cropped_array
def mask(image,x_origin,y_origin,x_length,y_length):
    x, y = image.shape
    one_array = np.ones([x, y])
    one_array[x_origin:x_origin+x_length+1, y_origin:y_origin+y_length+1] = 0
    masked_array = one_array * image
    return masked_array
def SpectralFFT(array):
    # Take the fourier transform of the image.
    F1 = fftpack.fft2(array)

    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift(F1)

    # Calculate a 2D power spectrum
    psd2D = np.abs(F2) ** 2

    # Calculate the radially averaged 1D power spectrum
    spectra = radial_profile2(psd2D)

    return spectra
def hann_SpectralFFT(array):
    wimage_gray = array * window('hann', array.shape)
    windowed_array_spectrum = SpectralFFT(wimage_gray)
    return windowed_array_spectrum

image = plt.imread(image_loc)
image_gray = normalise(rgb2gray(image))

x, y = image_gray.shape

# if > data.shape
#     print('No')

ox = rectangle_x_origin
oy = rectangle_y_origin
rx = rectangle_x_length
ry = rectangle_y_length

c_x = [ox,ox+rx,ox+rx,ox,ox] #make a set of coordinates to draw the rectangle on the image
c_y = [oy,oy,oy+ry,oy+ry,oy]

if crop_mask == 0:
    cm_array = crop(image_gray,rectangle_x_origin,rectangle_y_origin,rectangle_x_length,rectangle_y_length)
else:
    cm_array = mask(image_gray,rectangle_x_origin,rectangle_y_origin,rectangle_x_length,rectangle_y_length)

if segment == 1:
    thresh = threshold_otsu(cm_array[rectangle_x_origin:rectangle_x_origin+rectangle_x_length+1, rectangle_y_origin:rectangle_y_origin+rectangle_y_length+1])
    cm_array = cm_array > thresh

py.figure(1)
py.clf()
py.imshow(cm_array, cmap=py.cm.Greys)
py.plot(c_y, c_x,'tab:orange', ms=10)
py.show()


psd1D = SpectralFFT(cm_array)

bin = 179  #2k=179

if crop_mask == 0:
    horz = np.linspace(0, np.sqrt((rectangle_x_length / 2) ** 2 + (rectangle_y_length / 2) ** 2),
                   bin) / (image_size * 1000)
else:
    horz = np.linspace(0, np.sqrt((x / 2) ** 2 + (y / 2) ** 2),
                   bin) / (image_size * 1000)  # Use Pythagoras to work out the furthest radius on the image
                    # change this to size of rectangle for crop


py.figure(2)
py.clf()
py.semilogy(horz[:255], psd1D[:255], label='FFT', lw=0.5)
py.ylim([1e6, 1e11])
py.yticks([])
py.ylabel('Intensity (a.u.)')
py.xlabel('Wave vector /$nm^{-1}$')

# Inset 2D FFT in figure?
# Dots over lines?
# Make a separate function for windowed FFT
# See how much cropping actually affects the FFT
# See how much segmentation actually affects the FFT


py.show()

py.figure(3)  # WINDOWING TEST
py.clf()
py.semilogy(horz[:255], psd1D[:255], label='FFT', lw=0.5)
py.semilogy(horz[:255], hann_SpectralFFT(cm_array), label='wFFT', lw=0.5)
py.ylim([1e6, 1e11])
py.yticks([])
py.ylabel('Intensity (a.u.)')
py.xlabel('Wave vector /$nm^{-1}$')
py.legend(loc="best", fontsize='xx-small')

py.show()

py.figure(4)  # true cropping vs zeroing
py.clf()
horz = np.linspace(0, np.sqrt((x / 2) ** 2 + (y / 2) ** 2),
                   bin) / (image_size * 1000)
py.semilogy(horz, psd1D, label='FFT', lw=0.5)

cm2 = cm_array[rectangle_x_origin:rectangle_x_origin+rectangle_x_length,rectangle_y_origin:rectangle_y_origin+rectangle_y_length]
cpsd1D = SpectralFFT(cm2)
bin = 106 # 2k =106
horz2 = np.linspace(0, np.sqrt((rectangle_x_length / 2) ** 2 + (rectangle_y_length / 2) ** 2),
                   bin) / (image_size * 1000)
py.semilogy(horz2, cpsd1D, label='cFFT', lw=0.5)

# py.ylim([1e6, 1e11])
py.yticks([])
py.ylabel('Intensity (a.u.)')
py.xlabel('Wave vector /$nm^{-1}$')
py.legend(loc="best", fontsize='xx-small')
py.show()


if comparison == 1:
    # calculate corners of second rectangle
    ox2 = rectangle2_x_origin
    oy2 = rectangle2_y_origin
    rx2 = rectangle2_x_length
    ry2 = rectangle2_y_length

    c_x2 = [ox2, ox2 + rx2, ox2 + rx2, ox2, ox2]  # make a set of coordinates to draw the rectangle on the image
    c_y2 = [oy2, oy2, oy2 + ry2, oy2 + ry2, oy2]

    cm_array2 = mask(image_gray, rectangle2_x_origin, rectangle2_y_origin, rectangle2_x_length, rectangle2_y_length)

    if segment == 1:
        thresh2 = threshold_otsu(cm_array2[rectangle2_x_origin:rectangle2_x_origin + rectangle2_x_length + 1,
                                rectangle2_y_origin:rectangle2_y_origin + rectangle2_y_length + 1])
        cm_array2 = cm_array2 > thresh2

    py.figure(5)
    py.clf()
    py.imshow(cm_array2, cmap=py.cm.Greys)
    py.plot(c_y2, c_x2, 'tab:orange', ms=10)
    py.show()

    psd1D2 = SpectralFFT(cm_array2)
    horz_comp = np.linspace(0, np.sqrt((x / 2) ** 2 + (y / 2) ** 2),
                   len(psd1D2)) / (image_size * 1000)

    py.figure(6)  # 2 different regions (crop or mask) vs
    py.clf()
    region1=70
    region2=70
    py.semilogy(horz2[:region1:2], cpsd1D[:region1:2], 'ko', label='FFT Cropped Region', ms=5)
    py.semilogy(horz_comp[:region2:2], psd1D2[:region2:2], 'bs', label='FFT Masked Region', ms=5)
    py.yticks([])
    py.ylabel('Intensity (a.u.)')
    py.xlabel('Wave vector /$nm^{-1}$')
    py.legend(loc="best", fontsize='xx-small')
    py.show()

# Using 1x4 image:
# Change in vector and frequency observed! Nice!
# change to dots and squares
# remove a chunk of the data points
# crop the fall-off
# boost the font size



# 'ko' for black dots
# 'bs' is blue squares