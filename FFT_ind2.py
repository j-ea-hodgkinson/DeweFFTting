# Code that takes in a singular image, and can return the radially averaged
# 1D FFT spectrum of two squares of the same size.
# Used mainly for oxide layer images, towards the bottom is the option of comparing
# a cropped to a mask region with two different rectangles.

# Import the usual suspects
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import pylab as py
from skimage.filters import window
import statsmodels.api as sm


# Import every from the other FFT code, half of these aren't used
def rgb2gray(rgb):  # Define a function that converts PNGs to grayscale arrays
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
def normalise(array):  # Set an image numpy array's values to be between 0 and 1
    norm_array = (array-np.min(array))\
                         / (np.max(array)-np.min(array))
    return norm_array
def radial_profile2(data, center=None, binning=1):  # Radial profile calculator
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

img_name = 'Si100SiO2_1_75Au0004_0_Fourier'  # Take a file name from the list, or insert your file name here
file_name = img_name + '.png'
# Si100SiO2_1_75Au0004_0
# 80

# label = '1.75 g/l'


run_no = 1  #  appends a number to image, useful tool for when you don't want to overwrite old data
image_size = 80  # provide the image size in microns
# BLACK IS SQUARE 1, PUT IT ON THE OXIDE
# ORANGE IS SQUARE 2, PUT IT ON THE SILICON
# look_good = 1  # change to 1 when you're happy with the square locations, black is square 1, orange is square 2
square_1_origin = [1, 1] # 55,60-470 increments of 82,  142 224 306 388
# square_2_origin = [100,0] # 55,0 size:50
square_size = 512  # size of square in pixels
#
# nudes = 0  # set to 1 if you are in Oxide Data/Nudes



# Make the parameters for use in drawing
ox1 = square_1_origin[0]
oy1 = square_1_origin[1]
rx1 = square_size
ry1 = square_size
c_x1 = [ox1, ox1+rx1, ox1+rx1, ox1, ox1]  # make a set of coordinates to draw the rectangle on the image
c_y1 = [oy1, oy1, oy1+ry1, oy1+ry1, oy1]


# ox2 = square_2_origin[0]
# oy2 = square_2_origin[1]
# rx2 = square_size
# ry2 = square_size
# c_x2 = [ox2, ox2+rx2, ox2+rx2, ox2, ox2]  # make a set of coordinates to draw the rectangle on the image
# c_y2 = [oy2, oy2, oy2+ry2, oy2 + ry2, oy2]


# Find the image directory
# if nudes == 1:
#     image_loc = 'Ind Data/' + file_name
# else:
image_loc = 'Ind Data/' + file_name
# Load the image, then grayscale and normalise the pixel values
image = plt.imread(image_loc)
image_gray = normalise(rgb2gray(image))
# Returns the image shape, used sometimes
x, y = image_gray.shape

# Plot the image with the 2 squares drawn on it
# py.figure(1)
# py.clf()
# py.imshow(image_gray, cmap=py.cm.Greys)
# py.plot(c_y1, c_x1, 'black', ms=10)
# py.plot(oy1, ox1, 'kx', ms=10)
# py.plot(c_y2, c_x2, 'tab:orange', ms=10)
# py.plot(oy2, ox2, 'kx', ms=10)
# fig_sav_loc = 'res/oxide/comparesquares/' + img_name + 'locns_' + str(run_no) + '.png'
# py.savefig(fig_sav_loc, dpi=300)
# py.show()

# Crop to the squares
crop_square1_array = image_gray[ox1:ox1+rx1, oy1:oy1+ry1]
# crop_square2_array = image_gray[ox2:ox2+rx2, oy2:oy2+ry2]

# Display the returned cropped region
# py.figure(2)
# py.clf()
# py.subplot(1, 2, 1)
# py.imshow(crop_square1_array, cmap=py.cm.Greys)
# py.subplot(1, 2, 2)
# py.imshow(crop_square2_array, cmap=py.cm.Greys)
# py.show()

# Quits out if you aren't ready yet ie look_good=0
# if look_good == 0:
#     exit

# Save the raw 2D FFT of the images for the paper
# Square 1
F1 = fftpack.fft2(crop_square1_array)
F2 = fftpack.fftshift(F1)
cm_2d = np.abs(F2) ** 2
s1_sav_loc = 'Ind Data/res/' + img_name + '_square1_' + str(run_no) + '.png'
py.imsave(s1_sav_loc, np.log(cm_2d), dpi=300, cmap='gray')

# Square 2
# F1 = fftpack.fft2(crop_square2_array)
# F2 = fftpack.fftshift(F1)
# cm2_2d = np.abs(F2) ** 2
# s2_sav_loc = 'res/oxide/comparesquares/' + img_name + 'square2_' + str(run_no) + '.png'
# py.imsave(s2_sav_loc, np.log(cm2_2d), dpi=300, cmap='rainbow')


# Solve for the 1D power spectra
psd1D_1 = hann_SpectralFFT(crop_square1_array)
# psd1D_2 = SpectralFFT(crop_square2_array)

# Calculate the wavevector to plot on the x-axis, using the provided image size
ratio = square_size / x
horz_1 = np.linspace(0, np.sqrt((square_size / 2) ** 2 + (square_size / 2) ** 2),
                   len(psd1D_1)) / (image_size * ratio)
# horz_2 = np.linspace(0, np.sqrt((square_size / 2) ** 2 + (square_size / 2) ** 2),
#                    len(psd1D_2)) / (image_size * ratio * 1000)

# lowess = sm.nonparametric.lowess

# Display the 1D radially averaged FFT of the 2 squares on the same graph
py.figure(3)  # 2 different regions (crop or mask) vs
py.clf()
py.rcParams['font.size'] = '17'
region1 = int(np.floor(len(psd1D_1)/np.sqrt(2)))  # Use the ratio between the diagonal and side length to remove cusp
# region2 = int(np.floor(len(psd1D_2)/np.sqrt(2)))  # ' '
# py.axvline(0.21, color='orange')
# py.axvline(1.12, color='orange')
# py.axvline(1.66, color='orange')
py.axvline(0.21, color='k')
py.axvline(1.12, color='k')
py.axvline(1.66, color='k')
py.semilogy(horz_1[1:region1:1], psd1D_1[1:region1:1], 'tab:orange', linewidth=1)
# py.semilogy(horz_1[1:region1:1], psd1D_1[1:region1:1], 'md', label=label, ms=3)
# py.xlim(0, 0.0511)
# py.ylim(100000, 150000000)
# z = lowess(psd1D_1[1:region1:1], horz_1[1:region1:1], frac=1./5)
# py.semilogy(z[:, 0], z[:, 1], 'r')
# 10 point LOWESS
# py.semilogy(horz_2[2:region2:1], psd1D_2[2:region2:1], 's', color='orange', label='FFT Region 2', ms=5)
# py.yticks([])
py.ylabel('Intensity (a.u.)', fontsize=22)
py.xlabel('Wave vector ($\mu m^{-1}$)', fontsize=22)
# py.legend(loc="best", fontsize='xx-small')
fig_sav_loc = 'Ind Data/res/' + img_name + 'plot_' + str(run_no) + '.svg'
py.savefig(fig_sav_loc, dpi=300)
fig_sav_loc = 'Ind Data/res/' + img_name + 'plot_' + str(run_no) + '.png'
py.savefig(fig_sav_loc, dpi=300)
py.show()

# # Display the wave vector of the peak intensity
peak = horz_1[np.argmax(psd1D_1[1:])+1]
print('Black Square peak is ' + str(horz_1[np.argmax(psd1D_1[1:])+1]))
print(2*np.pi/peak)
# print('Orange Square peak is ' + str(horz_2[np.argmax(psd1D_2[1:])+1]))
# print('Delta = ' + str(horz_1[np.argmax(psd1D_1[1:])]-horz_2[np.argmax(psd1D_2[1:])]))
#
# # Calculate and display the area under the 1D radially averaged FFT ie the rms roughness squared
# area1 = np.trapz(psd1D_1[1:region1], x=horz_1[1:region1:1])
# area2 = np.trapz(psd1D_2[1:region2], x=horz_2[1:region2:1])
#
# print('Area under Oxide PSD is ' + str(area1))
# print('Area under Silicon PSD is ' + str(area2))
print(np.max(psd1D_1[1:region1:1]))
print(np.min(psd1D_1[1:region1:1]))