# Code that takes in a singular image, and can return the radially averaged
# 1D FFT spectrum using a rectangle as either a crop or mask.
# Used mainly for oxide layer images, towards the bottom is the option of comparing
# a cropped to a mask region with two different rectangles.

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


img_name = '1e_5_4x4_xt_2'  # Take a file name from the list
file_name = img_name + '.png'
# 1e_5_4x4_xt_2
# 2b_2.25_5x5_hs
# 2i_2.25_3x3_hs
# 2k_2.25_1x4_hs
# 'SiPatternedAu1_750007'

image_size = 7.6  # provide the image size in microns 2k is 8, SiPat is 7.6, 1e 7.8
crop_mask = 0  # crop if 0, mask if 1
segment = 0  # 0 to FFT the grayscale image, 1 to binarise the image first
comparison = 1  # makes a final 2 figures of both two rectangles and their respective FFTs, assumes c_m = 0

# Choose the location and size of the rectangle
rectangle_x_length = 270
rectangle_y_length = 270
rectangle_x_origin = 100
rectangle_y_origin = 120
#2k use 180, 300, 250, 0

if comparison == 1:  # enter the coordinates of a second rectangle you'll use for masking
    rectangle2_x_length = 350
    rectangle2_y_length = 350
    rectangle2_x_origin = 70
    rectangle2_y_origin = 70
    # 2k uses, 220, 390, 230, 0

image_loc = 'Oxide Data/Test Set/' + file_name

def rgb2gray(rgb):  # Define a function that converts PNGs to grayscale arrays
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
def normalise(array):  # Set an image numpy array's values to be between 0 and 1
    norm_array = (array-np.min(array))\
                         / (np.max(array)-np.min(array))
    return norm_array
def radial_profile2(data, center=None, binning=2):  # Radial profile calculator
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
def crop(image,x_origin,y_origin,x_length,y_length):  # Set all values outside the rectangle to 0
    x, y = image.shape
    zero_array = np.zeros([x, y])
    zero_array[x_origin:x_origin+x_length+1, y_origin:y_origin+y_length+1] = 1
    cropped_array = zero_array * image
    return cropped_array
def mask(image,x_origin,y_origin,x_length,y_length):  # Set all values inside the rectangle to 0
    x, y = image.shape
    one_array = np.ones([x, y])
    one_array[x_origin:x_origin+x_length+1, y_origin:y_origin+y_length+1] = 0
    masked_array = one_array * image
    return masked_array
def SpectralFFT(array):  # Calculate the radially averaged 1D FFT spectrum
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
def hann_SpectralFFT(array):  # Calculate the radially averaged 1D FFT spectrum of a Hann windowed image
    wimage_gray = array * window('hann', array.shape)
    windowed_array_spectrum = SpectralFFT(wimage_gray)
    return windowed_array_spectrum

# Load the image, then grayscale and normalise the pixel values
image = plt.imread(image_loc)
image_gray = normalise(rgb2gray(image))

# Returns the image shape, used often
x, y = image_gray.shape

# if > data.shape
#     print('No')

# Make the parameters shorter to stop me typing so much
ox = rectangle_x_origin
oy = rectangle_y_origin
rx = rectangle_x_length
ry = rectangle_y_length

c_x = [ox,ox+rx,ox+rx,ox,ox] #make a set of coordinates to draw the rectangle on the image
c_y = [oy,oy,oy+ry,oy+ry,oy]

# Crops or masked image array based on selection
if crop_mask == 0:
    cm_array = crop(image_gray,rectangle_x_origin,rectangle_y_origin,rectangle_x_length,rectangle_y_length)
else:
    cm_array = mask(image_gray,rectangle_x_origin,rectangle_y_origin,rectangle_x_length,rectangle_y_length)

# Segment the image using Otsu's threshold if asked to
if segment == 1:
    thresh = threshold_otsu(cm_array[rectangle_x_origin:rectangle_x_origin+rectangle_x_length+1, rectangle_y_origin:rectangle_y_origin+rectangle_y_length+1])
    cm_array = cm_array > thresh

# Display the cropped/masked region
py.figure(1)
py.clf()
py.imshow(cm_array, cmap=py.cm.Greys)
py.plot(c_y, c_x,'tab:orange', ms=10)
py.show()

# Solve!
psd1D = SpectralFFT(cm_array)

# Set this to be the length of the resulting psd1D
# bin = 179  #2k=179

# Calculate the wavevector to plot on the x-axis, using the provided image size
if crop_mask == 0:
    horz = np.linspace(0, np.sqrt((rectangle_x_length / 2) ** 2 + (rectangle_y_length / 2) ** 2),
                   len(psd1D)) / (image_size * 1000)
else:
    horz = np.linspace(0, np.sqrt((x / 2) ** 2 + (y / 2) ** 2),
                   len(psd1D)) / (image_size * 1000)  # Use Pythagoras to work out the furthest radius on the image
                    # change this to size of rectangle for crop

# Plot the semi-log of the spectrum, [:X] is used to crop out the cusp on the FFT created
# by the degeneracy at the edges of the radius
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

# Tests the effect of using cropping to the perimeter of the rectangle instead of zeroing
py.figure(4)
py.clf()
bin = 106
horz = np.linspace(0, np.sqrt((x / 2) ** 2 + (y / 2) ** 2), len(psd1D)) / (image_size * 1000)
py.semilogy(horz, psd1D, label='FFT', lw=0.5)

#Crops using this line
cm2 = cm_array[rectangle_x_origin:rectangle_x_origin+rectangle_x_length,rectangle_y_origin:rectangle_y_origin+rectangle_y_length]
cpsd1D = SpectralFFT(cm2)
bin = 106 # 2k =106
ratio = (np.sqrt((rectangle_x_length / 2) ** 2 + (rectangle_y_length / 2) ** 2)) / np.sqrt((x / 2) ** 2 + (y / 2) ** 2)
horz2 = np.linspace(0, np.sqrt((rectangle_x_length / 2) ** 2 + (rectangle_y_length / 2) ** 2),len(cpsd1D)) / (image_size * ratio * 1000)
py.semilogy(horz2, cpsd1D, label='cFFT', lw=0.5)

# py.ylim([1e6, 1e11])
py.yticks([])
py.ylabel('Intensity (a.u.)')
py.xlabel('Wave vector /$nm^{-1}$')
py.legend(loc="best", fontsize='xx-small')
py.show()

# Comparing a crop to a mask of a different size
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

    cm_array2 = normalise(cm_array2)
    psd1D2 = SpectralFFT(cm_array2)
    horz_comp = np.linspace(0, np.sqrt((x / 2) ** 2 + (y / 2) ** 2),
                   len(psd1D2)) / (image_size * 1000)


    py.figure(6)  # 2 different regions (crop or mask) vs
    py.clf()
    py.rcParams['font.size'] = '20'
    region1=69 # modify by hand to crop off the degenerate cusp
    region2=70 # ' '
    py.semilogy(horz2[1:region1:2], cpsd1D[1:region1:2], 'ko', label='FFT Cropped Region', ms=5)
    py.semilogy(horz_comp[2:region2:2], psd1D2[2:region2:2], 's' ,color='orange', label='FFT Masked Region', ms=5)
    py.yticks([])
    py.ylabel('Intensity (a.u.)',fontsize=22)
    py.xlabel('Wave vector ($nm^{-1}$)',fontsize=22)
    # py.legend(loc="best", fontsize='xx-small')
    fig_sav_loc = 'res/oxide/comparison/' + img_name + 'plot.svg'
    py.savefig(fig_sav_loc, dpi=300)
    py.show()

# Using 1x4 image:
# Change in vector and frequency observed! Nice!
# change to dots and squares
# remove a chunk of the data points
# crop the fall-off
# boost the font size



# 'ko' for black dots
# 'bs' is blue squares

    # Save the 2D FFTs for paper

    # cm_array
    F1 = fftpack.fft2(cm_array)
    F2 = fftpack.fftshift(F1)
    cm_2d = np.abs(F2) ** 2
    crop_sav_loc = 'res/oxide/comparison/' + img_name + '2dFFTcrop.png'
    py.imsave(crop_sav_loc, np.log(cm_2d), dpi=300, cmap='rainbow')

    # cm_array2
    F1 = fftpack.fft2(cm_array2)
    F2 = fftpack.fftshift(F1)
    cm2_2d = np.abs(F2) ** 2
    mask_sav_loc = 'res/oxide/comparison/' + img_name + '2dFFTmask.png'
    py.imsave(mask_sav_loc, np.log(cm2_2d), dpi=300, cmap='rainbow')


# Used for SiPatternedAu1 in particular, cause I wanted the whole outside area
# image_loc = 'Oxide Data/Test Set/SiPatternedAu1_750007cut2.png'
# image = plt.imread(image_loc)
# image_gray = normalise(rgb2gray(image))
# F1 = fftpack.fft2(image_gray)
# F2 = fftpack.fftshift(F1)
# cm2_2d = np.abs(F2) ** 2
# mask_sav_loc = 'res/oxide/comparison/' + img_name + '2dFFTcrop.png'
# py.imsave(mask_sav_loc, np.log(cm2_2d), dpi=300, cmap='rainbow')
#
# cpsd1D2 = SpectralFFT(image_gray)
# ratio = 281/512
# horz2 = np.linspace(0, np.sqrt((281 / 2) ** 2 + (281 / 2) ** 2),len(cpsd1D)) / (image_size * ratio * 1000)
#
# py.semilogy(horz2[:region1:2], cpsd1D[:region1:2], 'ko', label='FFT Cropped Region', ms=5)
#
# image_loc = 'Oxide Data/Test Set/SiPatternedAu1_750007cut.png'
# image = plt.imread(image_loc)
# image_gray = normalise(rgb2gray(image))
# F1 = fftpack.fft2(image_gray)
# F2 = fftpack.fftshift(F1)
# cm2_2d = np.abs(F2) ** 2
# mask_sav_loc = 'res/oxide/comparison/' + img_name + '2dFFTmask.png'
# py.imsave(mask_sav_loc, np.log(cm2_2d), dpi=300, cmap='rainbow')
#
# psd1D2 = SpectralFFT(image_gray)
# py.semilogy(horz_comp[:region2:2], psd1D2[:region2:2], 's' ,color='orange', label='FFT Masked Region', ms=5)
# py.savefig(fig_sav_loc, dpi=300)
# py.show()