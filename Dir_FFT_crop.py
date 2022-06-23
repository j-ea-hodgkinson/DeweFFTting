# Finds all PNGs in a directory and solves the radially averaged 1D FFT power spectra for each, saving the peak parameters to a table

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

# Find all PNGs in the target directory
dir = 'Data/Test Data'  # / 'Test Data'
files = os.listdir(dir)
files_png = [i for i in files if i.endswith('.png')]

test_windows = 1  # Windows the images using different methods from Hann beforehand if set to 1
isolated = 0  # Saves graphs of a select few PNGs

line = 0.5  # Line thickness for graphs
marker = 3  # Marker size for graphs

# Define a function that converts PNGs to grayscale arrays
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

# Define a function that calculates the radially averaged profile of a 2D power spectrum
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

def normalise(array):  # Set an image numpy array's values to be between 0 and 1
    norm_array = (array-np.min(array))\
                         / (np.max(array)-np.min(array))
    return norm_array


# Write a file name RegEx that all file names take to extract the data from the file name
re_input = re.compile('(?P<run_no>\d+)_outfile_C(?P<C>\d+)_kT(?P<kT>\d+)_MR1_mu0(?P<mu0>\d+)_muf15_vus75_sig01_L1024_m(?P<m>\d+)_IC.png')

# Define a table for the values
dataframe = pd.DataFrame(columns=['File Name', 'Run No.', 'C', 'KT', 'mu0', 'm', 'FFT peak index', 'FFT peak wavevector', 'FFT peak prominence', 'FFT peak intensity'])
# line 229

# Just some loop pieces
graph_count = 0
na_count = 0

# WHITE NOISE CORRECTION - NOT USED
# Make a grayscale white noise image to return the 1D power spectrum, the process here is explained elsewhere
# Probably should smooth this, change it so values can only be 0, 0.65905 or 1
wn_image = normalise(np.random.uniform(0, 1, [1024, 1024]))
F1 = fftpack.fft2(wn_image)
F2 = fftpack.fftshift( F1 )
psd2D = np.abs( F2 )**2
wn_psd1D = radial_profile2(psd2D)

# Loop through all PNGs
for j in files_png:
    image_name = j

    # Use the RegEx to save the parameters in the file name
    match = re_input.match(image_name)
    run_no = match.group('run_no')
    C = match.group('C')
    kT = match.group('kT')
    mu0 = match.group('mu0')
    m = match.group('m')
    # dataframe.loc[files_png.index(j)] = [image_name] + [run_no] + [C] + [kT] + [mu0] + [m]

    # Used for testing just a handful of images in test set
    # if graph_count in {0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
    #                    1600} or run_no == '12254': # delete this and unindent before next 'if graph count statement' to run on all images


    # Load the image
    image_loc = r'Data/' + image_name
    image = plt.imread(image_loc)
    image_gray = normalise(rgb2gray(image))

    # Take the fourier transform of the image.
    F1 = fftpack.fft2(image_gray)

    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift(F1)

    # Calculate a 2D power spectrum
    psd2D = np.abs(F2) ** 2

    # Calculate the radially averaged 1D power spectrum
    psd1D = radial_profile2(psd2D)

    # Window the image using Hann, calculate the 1D spectrum as before
    wimage_gray = image_gray * window('hann', image_gray.shape)  # hamming, blackman, blackmanharris
    wF1 = fftpack.fft2(wimage_gray)
    wF2 = fftpack.fftshift(wF1)
    wpsd2D = np.abs(wF2) ** 2
    wpsd1D = radial_profile2(wpsd2D)

    swpsd1D = uniform_filter1d(wpsd1D[1:], 15)  # smooth the spectra with a moving average function, removing the first value as it's often a large peak
    # Find the peak index in the spectrum, using find_peaks
    speaks, properties = signal.find_peaks(np.log(swpsd1D), prominence=1)

    if test_windows == 1:
        # Window using 3 different windowing functions
        hammwimage_gray = image_gray * window('hamming', image_gray.shape)  # hamming, blackman, blackmanharris
        hammwF1 = fftpack.fft2(hammwimage_gray)
        hammwF2 = fftpack.fftshift(hammwF1)
        hammwpsd2D = np.abs(hammwF2) ** 2
        hammwpsd1D = radial_profile2(hammwpsd2D)

        bmwimage_gray = image_gray * window('blackman', image_gray.shape)  # hamming, blackman, blackmanharris
        bmwF1 = fftpack.fft2(bmwimage_gray)
        bmwF2 = fftpack.fftshift(bmwF1)
        bmwpsd2D = np.abs(bmwF2) ** 2
        bmwpsd1D = radial_profile2(bmwpsd2D)

        bmhwimage_gray = image_gray * window('blackmanharris', image_gray.shape)  # hamming, blackman, blackmanharris
        bmhwF1 = fftpack.fft2(bmhwimage_gray)
        bmhwF2 = fftpack.fftshift(bmhwF1)
        bmhwpsd2D = np.abs(bmhwF2) ** 2
        bmhwpsd1D = radial_profile2(bmhwpsd2D)

        ftwimage_gray = image_gray * window('flattop', image_gray.shape)  # hamming, blackman, blackmanharris
        ftwF1 = fftpack.fft2(ftwimage_gray)
        ftwF2 = fftpack.fftshift(ftwF1)
        ftwpsd2D = np.abs(ftwF2) ** 2
        ftwpsd1D = radial_profile2(ftwpsd2D)


    if not speaks:  # If no peak is found, return N/A for all parameters and uptick N/A count
        fft_peak = 'NaN'
        fft_wavevector = 'NaN'
        fft_prominence = 'NaN'
        fft_height = 'NaN'
        na_count += 1

    else:  # If peaks are found, calculate the location, height and spread of the peak
        fft_peak = speaks[0]

        x, y = image_gray.shape
        horz = np.linspace(0, np.sqrt((x / 2) ** 2 + (y / 2) ** 2),
                           362) / 20  # Use Pythagoras to work out the furthest radius on the image
        fft_wavevector = horz[fft_peak]

        fft_prominence = properties['prominences'][0]
        fft_height = np.log(swpsd1D)[fft_peak]

    # Show graphs of the test set
    if graph_count in {0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600} or run_no == '37108':
        # if run_no in {12254}:
        progress = str(graph_count) + ' of 1630 done.'
        print(progress)

        # Calculate the x-axis in per-micron
        x, y = image_gray.shape
        horz = np.linspace(0, np.sqrt((x / 2) ** 2 + (y / 2) ** 2), 362) / 1024  # Use Pythagoras to work out the furthest radius on the image
        # 20 is used as the image is assumed to be 20x20 microns, or 1024 for case of pixels

        # Display a semi-log plot of the 1D spectrum
        py.figure(1)
        py.clf()
        py.semilogy(horz[:255], psd1D[:255], label='FFT', lw=line)
        py.ylim([1e5, 1e10])

        # You will need to change the ylim and the limiting index on the semilog plot dependent on your own data
        # 255 works for 1024x1024 images, as beyond that you're calculating power for radii off the image, resulting in a cusp



        # sav_loc = r'res/' + j.replace('.png', '')

        # Display a semi-log plot of the 1D spectrum with windowing, as well as the peak location
        py.semilogy(horz[:255], wpsd1D[:255], label='FFT w/ Hanning windowing', lw=line)
        peaks, properties = signal.find_peaks(np.log(wpsd1D), prominence=1)
        # py.plot(horz[peaks], wpsd1D[peaks], 'X', label='Unsmoothed peaks', ms=marker) # Uncomment later
        # py.semilogy(horz, psd1D-wn_psd1D, label='FFT w/ white noise correction', lw=line)
        # py.semilogy(horz, wn_psd1D, label='White noise FFT', lw=line)

        if test_windows == 1:
            # Plot other windows
            py.semilogy(horz[:255], hammwpsd1D[:255], label='FFT w/ Hamming windowing', lw=line)
            py.semilogy(horz[:255], bmwpsd1D[:255], label='FFT w/ Blackman windowing', lw=line)
            py.semilogy(horz[:255], bmhwpsd1D[:255], label='FFT w/ Blackman-Harris windowing', lw=line)
            py.semilogy(horz[:255], ftwpsd1D[:255], label='FFT w/ Flat-top windowing', lw=line)

        # Combinations of other methods
        # swpsd1D = uniform_filter1d(wpsd1D, 15) # smooth the spectra with a moving average function
        # py.semilogy(horz[1:255], swpsd1D[1:255], '--', label='FFT w/ Hanning windowing & smoothing', lw=line) # Uncomment later
        # speaks, properties = signal.find_peaks(np.log(swpsd1D), prominence=1)
        # py.plot(horz[speaks+1], swpsd1D[speaks], 'X', label='Smoothed peaks', ms=marker) # Uncomment later # 1 added to horz index to account for smoothing removing the first value


        # py.xlabel('Wave vector /$\u03BCm^{-1}$')
        py.xlabel('Wave vector /pixels$^{-1}$')
        py.ylabel('Power Spectrum')
        py.legend(loc="best", fontsize='xx-small')

        # Save the resulting image
        sav_loc = r'res/WIN_SPECTRA_' + run_no + '.png'
        py.savefig(sav_loc, dpi=300)
        sav_loc = r'res/WIN_SPECTRA_' + run_no + '.svg'
        py.savefig(sav_loc, dpi=300)

        if isolated == 1:
            py.figure(2)
            py.clf()
            py.ylim([1900000, 1e9])

            py.semilogy(horz[:255], wpsd1D[:255], label='FFT w/ Hanning windowing', lw=0.75)
            py.semilogy(horz[1:255], swpsd1D[1:255], 'k--', label='FFT w/ Hanning windowing & smoothing', lw=line)
            if speaks:
                py.plot(horz[speaks + 1], swpsd1D[speaks], 'kx', label='Peak = ' + str(float(np.round(horz[speaks+1], 2))) + ' /pixel$^{-1}$', ms=10)

            # py.xlabel('Wave vector /$\u03BCm^{-1}$')
            py.xlabel('Wave vector /pixels$^{-1}$')
            py.ylabel('Power Spectrum')
            py.legend(loc="best", fontsize='xx-small')

            sav_loc_iso = r'res/iso/ISO_SPECTRA_' + run_no
            py.savefig(sav_loc_iso, dpi=300)
            sav_loc_iso = r'res/iso/ISO_SPECTRA_' + run_no + '.svg'
            py.savefig(sav_loc_iso, dpi=300)




    graph_count += 1
    dataframe.loc[files_png.index(j)] = [image_name] + [run_no] + [C] + [kT] + [mu0] + [m] + [fft_peak] + [fft_wavevector] + [fft_prominence] + [fft_height]

print(str(1630-na_count) + ' of 1630 found peaks.')  # Print how many images could find a peak in the FFT
dataframe.to_csv(r'res/output2.csv') #Saves the new data to a directory, comment out if you're doing select images

