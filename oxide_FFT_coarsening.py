# Derivative of oxide_FFT.py
# Used to study two particular images of the same location on a surface 5 months apart

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
import imageio


file_name_0m = 'Si111_0_75_HF3_0004.png'
file_name_5m = 'SiPatternedAu0_750010.png'

image_size = 6  # provide the image size in microns
segmented = 0  # 0 to FFT the grayscale image, 1 to binarise the image first

# ox = rectangle_x_origin
# oy = rectangle_y_origin
# rx = rectangle_x_length
# ry = rectangle_y_length

# Locations of the rectangles that the code uses as crops upon the two images
# o is the origin, r is the length

# 1st rectangle
ox1 = 80
oy1 = 75
rx1 = 170
ry1 = 150

# 2nd rectangle
ox2 = 75
oy2 = 250
rx2 = 140
ry2 = 70

# 3rd rectangle
ox3 = 250
oy3 = 300
rx3 = 100
ry3 = 100

# 4th rectangle
ox4 = 425
oy4 = 360
rx4 = 85
ry4 = 150


def segment(array, x_origin, y_origin, x_length, y_length):  # Segment an array into a binary array using Otsu's
    thresh = threshold_otsu(array[x_origin:x_origin + x_length + 1, y_origin:y_origin + y_length + 1])
    segmented_array = array > thresh
    return segmented_array
def corners(x_origin, y_origin, x_length, y_length):  # Returns x and y coordinates
    corner_x = [x_origin,x_origin+x_length,x_origin+x_length,x_origin,x_origin] #make a set of coordinates to draw the rectangle on the image
    corner_y = [y_origin,y_origin,y_origin+y_length,y_origin+y_length,y_origin]
    return corner_x, corner_y
def rgb2gray(rgb):  # Define a function that converts PNGs to grayscale arrays
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
def normalise(array):  # Set an image numpy array's values to be between 0 and 1
    norm_array = (array - np.min(array)) \
                 / (np.max(array) - np.min(array))
    return norm_array
def radial_profile2(data, center=None, binning=1):  # Calculates the radial average
    y, x = np.indices((data.shape))  # first determine radii of all pixels

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # radius of the image.
    r_max = np.max(r)
    bin_no = np.rint(r_max / binning).astype(int)

    ring_brightness, radius = np.histogram(r, weights=data, bins=bin_no)
    # plt.plot(radius[1:], ring_brightness)
    # plt.show()
    return ring_brightness
def crop(image, x_origin, y_origin, x_length, y_length):  # Set all values outside the rectangle to 0
    x, y = image.shape
    zero_array = np.zeros([x, y])
    zero_array[x_origin:x_origin + x_length + 1, y_origin:y_origin + y_length + 1] = 1
    cropped_array = zero_array * image
    return cropped_array
def mask(image, x_origin, y_origin, x_length, y_length):  # Set all values inside the rectangle to 0
    x, y = image.shape
    one_array = np.ones([x, y])
    one_array[x_origin:x_origin + x_length + 1, y_origin:y_origin + y_length + 1] = 0
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
def hann_SpectralFFT(array):  # Calculate the radially averaged 1D FFT spectrum but windows the array first
    wimage_gray = array * window('hann', array.shape)
    windowed_array_spectrum = SpectralFFT(wimage_gray)
    return windowed_array_spectrum
def SpectralCropFFT(array, x_origin, y_origin, x_length, y_length):  # Finds spectrum but true crops first
    cropped_array = array[x_origin:x_origin + x_length,y_origin:y_origin + y_length]
    cropped_array = segment(cropped_array) if segmented == 1 else cropped_array
    PSD = SpectralFFT(cropped_array)
    Horz = np.linspace(0, np.sqrt((rx1 / 2) ** 2 + (ry1 / 2) ** 2),
                           len(PSD)) / (image_size * 1000)
    return PSD, Horz

# Load the images, then converts them to normalised grayscale arrays
image1_loc = 'Oxide Data/Test Set/' + file_name_0m
image2_loc = 'Oxide Data/Test Set/' + file_name_5m
image1 = plt.imread(image1_loc)
image1_gray = normalise(rgb2gray(image1))
image2 = plt.imread(image2_loc)
image2_gray = normalise(rgb2gray(image2))

# Assumes the image are the same size, they should be
x, y = image1_gray.shape

# Display the images with the rectangles labelled and overlayed on top
py.figure(1)
py.clf()
py.subplot(1,2,1)
py.xticks([])
py.yticks([])
py.imshow(image1, cmap='afmhot')

cx1,cy1 = corners(ox1,oy1,rx1,ry1)
py.plot(cy1, cx1,'w', ms=10)
py.annotate('1',[oy1,ox1-5],color='w')

cx2,cy2 = corners(ox2,oy2,rx2,ry2)
py.plot(cy2, cx2,'w', ms=10)
py.annotate('2',[oy2,ox2-5],color='w')

cx3,cy3 = corners(ox3,oy3,rx3,ry3)
py.plot(cy3, cx3,'w', ms=10)
py.annotate('3',[oy3,ox3-5],color='w')

cx4,cy4 = corners(ox4,oy4,rx4,ry4)
py.plot(cy4, cx4,'w', ms=10)
py.annotate('4',[oy4,ox4-5],color='w')


py.subplot(1,2,2)
py.xticks([])
py.yticks([])
py.imshow(image2, cmap='afmhot')

cx1,cy1 = corners(ox1,oy1,rx1,ry1)
py.plot(cy1, cx1,'w', ms=10)
py.annotate('1',[oy1,ox1-5],color='w')

cx2,cy2 = corners(ox2,oy2,rx2,ry2)
py.plot(cy2, cx2,'w', ms=10)
py.annotate('2',[oy2,ox2-5],color='w')

cx3,cy3 = corners(ox3,oy3,rx3,ry3)
py.plot(cy3, cx3,'w', ms=10)
py.annotate('3',[oy3,ox3-5],color='w')

cx4,cy4 = corners(ox4,oy4,rx4,ry4)
py.plot(cy4, cx4,'w', ms=10)
py.annotate('4',[oy4,ox4-5],color='w')

py.show()

# Calculate the FFT for inside every rectangle provided
PSD_1_1, Horz_1_1 = SpectralCropFFT(image1_gray,ox1,oy1,rx1,ry1)
PSD_1_2, Horz_1_2 = SpectralCropFFT(image1_gray,ox2,oy2,rx2,ry2)
PSD_1_3, Horz_1_3 = SpectralCropFFT(image1_gray,ox3,oy3,rx3,ry3)
PSD_1_4, Horz_1_4 = SpectralCropFFT(image1_gray,ox4,oy4,rx4,ry4)
PSD_2_1, Horz_2_1 = SpectralCropFFT(image2_gray,ox1,oy1,rx1,ry1)
PSD_2_2, Horz_2_2 = SpectralCropFFT(image2_gray,ox2,oy2,rx2,ry2)
PSD_2_3, Horz_2_3 = SpectralCropFFT(image2_gray,ox3,oy3,rx3,ry3)
PSD_2_4, Horz_2_4 = SpectralCropFFT(image2_gray,ox4,oy4,rx4,ry4)


# Display figures comparing the FFTs for the regions in both images
# top left 1, bottom right 4
py.figure(2)
py.clf()
py.subplot(2,2,1)
py.semilogy(Horz_1_1, PSD_1_1, label='FFT_1_1', lw=0.5)
py.semilogy(Horz_2_1, PSD_2_1, label='FFT_2_1', lw=0.5)
py.yticks([])
py.ylabel('Intensity (a.u.)')
py.xlabel('Wave vector /$nm^{-1}$')

py.subplot(2,2,2)

py.semilogy(Horz_1_2, PSD_1_2, label='FFT_1_2', lw=0.5)
py.semilogy(Horz_2_2, PSD_2_2, label='FFT_2_2', lw=0.5)
py.yticks([])
py.ylabel('Intensity (a.u.)')
py.xlabel('Wave vector /$nm^{-1}$')

py.subplot(2,2,3)

py.semilogy(Horz_1_3, PSD_1_3, label='FFT_1_3', lw=0.5)
py.semilogy(Horz_2_3, PSD_2_3, label='FFT_2_3', lw=0.5)
py.yticks([])
py.ylabel('Intensity (a.u.)')
py.xlabel('Wave vector /$nm^{-1}$')

py.subplot(2,2,4)

py.semilogy(Horz_1_4, PSD_1_4, label='FFT_1_4', lw=0.5)
py.semilogy(Horz_2_4, PSD_2_4, label='FFT_2_4', lw=0.5)
py.yticks([])
py.ylabel('Intensity (a.u.)')
py.xlabel('Wave vector /$nm^{-1}$')

# Inset 2D FFT in figure?
# Dots over lines?
# Make a separate function for windowed FFT
# See how much cropping actually affects the FFT
# See how much segmentation actually affects the FFT


py.show()

# change to dots and squares
# remove a chunk of the data points
# crop the fall-off
# boost the font size

# 'ko' for black dots
#'bs' is blue squares

save_movie = 0

# Make an animated one that sweeps the images simultaneously with adjustable window sizes, 2x2 subplots
image_number = 2  # 1 for 0 months, 2 for 5 months
window_size = 50  # Size of the rectangle in pixels
animation_steps = 4  # How fast the rectangle moves

# Don't run this for too long!  50 win_size for 2 ani_steps made 2 GB of data!

sweep_count = np.floor(x/window_size)
image = image1_gray if image_number == 1 else image2_gray

# Starting location of animation
o_x = 0
o_y = 0
r_x = window_size
r_y = window_size
frame_number = 0
sweeps = 0

print('Making spectra for movie...')

# Animation starts
while sweeps < sweep_count + 1:
    if save_movie == 0:
        break

    while o_y+r_y<x: # Move rectangle RIGHT, saving a subplot of the FFT at every stop
        PSD, Horz = SpectralCropFFT(image,o_x,o_y,r_x,r_y)
        py.figure(3)
        py.clf()
        py.subplot(1, 2, 1)
        py.imshow(image, cmap='afmhot')
        c_x,c_y = corners(o_x,o_y,r_x,r_y)
        py.plot(c_y, c_x,'w', ms=10)
        py.xticks([])
        py.yticks([])
        py.xlim([0,x])
        py.ylim([0,y])
        py.subplot(1, 2, 2)
        py.semilogy(Horz, PSD, lw=0.5)
        py.yticks([])
        py.ylim([1e-1, 1e5])
        # py.ylabel('Intensity (a.u.)')
        py.xlabel('Wave vector /$nm^{-1}$')
        sub_sav_loc = r'res/oxide/movie/im' + str(image_number) + '_win' + str(window_size) + '_st' + str(animation_steps) + '_frame' + str(frame_number) + '.png'
        py.savefig(sub_sav_loc, dpi=150)

        frame_number+=1
        o_y+=animation_steps

    sweeps+=1
    if sweeps == sweep_count-1:
        break

    while o_x < (sweeps+1)*window_size:  #DOWN
        PSD, Horz = SpectralCropFFT(image,o_x,o_y,r_x,r_y)
        py.figure(3)
        py.clf()
        py.subplot(1, 2, 1)
        py.imshow(image, cmap='afmhot')
        c_x,c_y = corners(o_x,o_y,r_x,r_y)
        py.plot(c_y, c_x,'w', ms=10)
        py.xticks([])
        py.yticks([])
        py.xlim([0,x])
        py.ylim([0,y])
        py.subplot(1, 2, 2)
        py.semilogy(Horz, PSD, lw=0.5)
        py.yticks([])
        py.ylim([1e-1, 1e5])
        # py.ylabel('Intensity (a.u.)')
        py.xlabel('Wave vector /$nm^{-1}$')
        sub_sav_loc = r'res/oxide/movie/im' + str(image_number) + '_win' + str(window_size) + '_st' + str(animation_steps) + '_frame' + str(frame_number) + '.png'
        py.savefig(sub_sav_loc, dpi=150)

        frame_number+=1
        o_x+=animation_steps

    while o_y+r_y>r_y:  #LEFT
        PSD, Horz = SpectralCropFFT(image,o_x,o_y,r_x,r_y)
        py.figure(3)
        py.clf()
        py.subplot(1, 2, 1)
        py.imshow(image, cmap='afmhot')
        c_x,c_y = corners(o_x,o_y,r_x,r_y)
        py.plot(c_y, c_x,'w', ms=10)
        py.xticks([])
        py.yticks([])
        py.xlim([0,x])
        py.ylim([0,y])
        py.subplot(1, 2, 2)
        py.semilogy(Horz, PSD, lw=0.5)
        py.yticks([])
        py.ylim([1e-1, 1e5])
        # py.ylabel('Intensity (a.u.)')
        py.xlabel('Wave vector /$nm^{-1}$')
        sub_sav_loc = r'res/oxide/movie/im' + str(image_number) + '_win' + str(window_size) + '_st' + str(animation_steps) + '_frame' + str(frame_number) + '.png'
        py.savefig(sub_sav_loc, dpi=150)

        frame_number+=1
        o_y-=animation_steps

    sweeps+=1
    if sweeps == sweep_count-1:
        break

    while o_x<(sweeps+1)*window_size:  #DOWN
        PSD, Horz = SpectralCropFFT(image,o_x,o_y,r_x,r_y)
        py.figure(3)
        py.clf()
        py.subplot(1, 2, 1)
        py.imshow(image, cmap='afmhot')
        c_x,c_y = corners(o_x,o_y,r_x,r_y)
        py.plot(c_y, c_x,'w', ms=10)
        py.xticks([])
        py.yticks([])
        py.xlim([0,x])
        py.ylim([0,y])
        py.subplot(1, 2, 2)
        py.semilogy(Horz, PSD, lw=0.5)
        py.yticks([])
        py.ylim([1e-1, 1e5])
        # py.ylabel('Intensity (a.u.)')
        py.xlabel('Wave vector /$nm^{-1}$')
        sub_sav_loc = r'res/oxide/movie/im' + str(image_number) + '_win' + str(window_size) + '_st' + str(animation_steps) + '_frame' + str(frame_number) + '.png'
        py.savefig(sub_sav_loc, dpi=150)

        frame_number+=1
        o_x+=animation_steps

    print('Animation ' + str((sweeps/sweep_count)*100) + '% done.')

print('Frames done...')

if save_movie == 1:  # Makes the gif
    print('Saving gif...')
    images = []
    for index in range(frame_number):
        sub_sav_loc = r'res/oxide/movie/im' + str(image_number) + '_win' + str(window_size) + '_st' + str(
            animation_steps) + '_frame' + str(index) + '.png'
        images.append(imageio.imread(sub_sav_loc))
    gif_sav_loc = r'res/oxide/gif/' + 'im' + str(image_number) + '_win' + str(window_size) + '_st' + str(animation_steps) +'.gif'
    imageio.mimsave(gif_sav_loc, images, duration=0.2)

print('Gif saved!')

