# Code that looks at different runs of Monte Carlo simulations of gold nanoparticle
# dewetting.
# Returns a gif of how the radially averaged 1D FFT spectra change with the simulation.

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import re
import pylab as py
import imageio

# From the folder structure, select the type of pattern and the run number, edit these accordingly
selected_type = 3  # 0'Cellular', 1'Fingering', 2'Holes', 3'Islands', 4'Labyrinthine', 5'Worm-like'
selected_run = 2
remove_liquid = 1  # Make it so the liquid isn't present
save_movie = 0  # Save a gif of the coarsening, saving the subplots to a movie folder
image_size = 20  # Provide the image size in microns


paper = 0  # Used for plotting some PSDs on the same graph, for the thesis
if selected_type == 1:
    selected_frames = [420, 2920, 3820, 5500] # For use with paper
    graph_in = 3
elif selected_type == 3:
    selected_frames = [140,240,1080,1500]
    graph_in = 8
elif selected_type == 5:
    selected_frames = [140, 520, 720, 1000]
    graph_in = 8

height = 0

# Fingering 2: 420, 2920,3820, 5500, 3:230
# Islands 2: 140, 240, 1080, 1500, 8:230
# Worm-like 2: 140, 520, 720, 1000, 8
psd1D_paper = np.zeros((len(selected_frames),362)) # 362 is the number of values in the x array for the plot
i=-1 # Counter for this cause I got lazy

# Sets the thickness and sizes of elements in the plotsFFT_Frame_crop.py
line = 0.7
marker = 3

# Find the chosen folder
type = ['Cellular', 'Fingering', 'Holes', 'Islands', 'Labyrinthine', 'Worm like']
type_dir = r'Frame Data/' + type[selected_type] + '/' + str(selected_run) + '/'

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


# Define a function that sets an image numpy array's values to be between 0 and 1
def normalise(array):
    norm_array = (array-np.min(array))\
                         / (np.max(array)-np.min(array))
    return norm_array

# Look into the folder for all images that are PNGs
files = os.listdir(type_dir)
files_png = [i for i in files if i.endswith('.png')]

# Write a file name RegEx that all file names take to extract the data from the file name
re_input = re.compile('outfile_C(?P<C>\d+)_kT(?P<kT>\d+)_MR1_mu0(?P<mu0>\d+)_muf15_vus75_sig01_L1024_m(?P<m>\d+)_IC.png')

print('Sorting files...')
# Sort images in order of monte carlo steps, m
steps = list([])  # Make an empty array for sorting
for j in files_png:
    image_name = j
    match = re_input.match(image_name)
    m = match.group('m')
    steps.append(int(m))

steps_sort = np.argsort(steps)

print('Calculating power spectra...')
# In the calculated order, start the loop
for index in steps_sort:
    # Load the image location
    image_name = files_png[index]
    image_loc = type_dir + image_name

    # Use the RegEx to find the value of m from the file name
    match = re_input.match(image_name)
    m = match.group('m')

    # Load the image from location then convert it to grayscale then normalise
    image = plt.imread(image_loc)
    image_gray = normalise(rgb2gray(image))

    # Convert the array such that the liquid isn't presence, if requested
    if remove_liquid == 1:
        image_gray[image_gray > 0.9] = 0  # Sets the liquid layer to 0
        image_gray = (image_gray > 0.5).astype(int)  # Sets the nanoparticle layer to 1

    # Take the 2D Fourier transform of the image.
    F1 = fftpack.fft2(image_gray)

    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift(F1)

    # Calculate a 2D power spectrum
    psd2D = np.abs(F2) ** 2

    # Calculate the radially averaged 1D power spectrum
    psd1D = radial_profile2(psd2D)

    height = np.append(height, np.max([psd1D[5:255]]))

        # # Apply windowing to the image before running the FFT code again
        # wimage_gray = image_gray * window('hann', image_gray.shape)
        # wF1 = fftpack.fft2(wimage_gray)
        # wF2 = fftpack.fftshift(wF1)
        # wpsd2D = np.abs(wF2) ** 2
        # wpsd1D = radial_profile2(wpsd2D)
        #
        # swpsd1D = uniform_filter1d(wpsd1D[1:], 15)  # smooth the spectra with a moving average function, removing the first value as it's often a large peak
        # speaks, properties = signal.find_peaks(np.log(swpsd1D), prominence=1)


        # if not speaks:
        #     fft_peak = 'N/A'
        #     na_count += 1
        #
        # else:
        #     fft_peak = speaks[0]

    # if graph_count in {0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600} or run_no == '12254':
    #     # if run_no in {12254}:
    #     progress = str(graph_count) + ' of 1630 done.'
    #     print(progress)

    # Calculate the x-axis in per-micron
    x, y = image_gray.shape
    horz = np.linspace(0, np.sqrt((x / 2) ** 2 + (y / 2) ** 2), 362) / image_size  # Use Pythagoras to work out the furthest radius on the image

    # Plot the 1D power spectrum in a log plot
    py.figure(1)
    py.clf()
    py.semilogy(horz[:255], psd1D[:255], label='FFT', lw=line)



        # sav_loc = r'res/' + j.replace('.png', '')


        # py.semilogy(horz, wpsd1D, label='FFT w/ Hanning windowing', lw=line)
        # peaks, properties = signal.find_peaks(np.log(wpsd1D), prominence=1)
        ## py.plot(horz[peaks], wpsd1D[peaks], 'X', label='Unsmoothed peaks', ms=marker)
        ## py.semilogy(horz, psd1D-wn_psd1D, label='FFT w/ white noise correction', lw=line)
        ## py.semilogy(horz, wn_psd1D, label='White noise FFT', lw=line)


        # swpsd1D = uniform_filter1d(wpsd1D, 15) # smooth the spectra with a moving average function
        # py.semilogy(horz[1:], swpsd1D, '--', label='FFT w/ Hanning windowing & smoothing', lw=line)
        ## speaks, properties = signal.find_peaks(np.log(swpsd1D), prominence=1)
        ## py.plot(horz[speaks+1], swpsd1D[speaks], 'X', label='Smoothed peaks', ms=marker) # 1 added to horz index to account for smoothing removing the first value



    py.xlabel('Wavevector /$\u03BCm^{-1}$')
    py.ylabel('Power Spectrum')
    # py.legend(loc="best", fontsize='xx-small')

    # Save both the figure and grayscale image to the results folder res/frame/
    fig_sav_loc = r'res/frame/' + type[selected_type] + '_' + str(selected_run) + '_1DFFT_m' + m + '.png'
    img_sav_loc = r'res/frame/' + type[selected_type] + '_' + str(selected_run) + '_IMG_l' + str(remove_liquid) + '_m' + m + '.png'
    py.savefig(fig_sav_loc, dpi=300)
    py.imsave(img_sav_loc, image_gray, dpi=300, cmap=py.cm.gray)

    print(m)
    if paper==1 and int(m) in selected_frames:
        i+=1
        print('Appending...')
        psd1D_paper[i] = psd1D
        # py.figure(3)





    # Make and save subplots for the gif
    if save_movie == 1:
        py.figure(2)
        py.clf()
        py.subplot(1, 2, 1)
        py.imshow(image_gray, cmap=py.cm.gray)
        py.xticks([])
        py.yticks([])
        py.title('m = ' + m)
        py.subplot(1, 2, 2)
        py.semilogy(horz[:255], psd1D[:255], label='FFT', lw=line)
        py.ylim([1e6, 1e11])
        py.xlabel('Wavevector /$\u03BCm^{-1}$')
        # py.ylabel('Power Spectrum')
        sub_sav_loc = r'res/frame/movie/' + type[selected_type] + '_' + str(selected_run) + '_SUB_l' + str(remove_liquid) + '_m' + m + '.png'
        py.savefig(sub_sav_loc, dpi=300)

if paper == 1:
    py.figure(3)
    py.clf()
    i = 0

    #green triangle, red square, blue circle, purple diamond
    markers_to_use = ['g^', 'rs', 'bo', 'md']
    lines_to_use = ['g','r','b','m']

    for m in selected_frames:
        psd1D = psd1D_paper[i]
        label = str(m) + ' Steps'
        f=(image_size / 1024)
        py.semilogy(f*horz[graph_in:255:2], psd1D[graph_in:255:2], lines_to_use[i], lw=line)
        py.semilogy(f*horz[graph_in:255:2], psd1D[graph_in:255:2], markers_to_use[i], ms=1, label=label, lw=line)
        i+=1

    py.xlabel('Wavevector /pixels$^{-1}$')
    py.ylabel('Intensity (a.u.)')
    py.yticks([])
    py.legend()
    sub_sav_loc = r'res/frame/paper/' + type[selected_type] + '_' + str(selected_run) + '_SUB_l' + str(
        remove_liquid) + '_m' + str(m) + '.png'
    sub_sav_loc_eps = r'res/frame/paper/' + type[selected_type] + '_' + str(selected_run) + '_SUB_l' + str(
        remove_liquid) + '_m' + str(m) + '.svg'
    py.savefig(sub_sav_loc, dpi=300)
    py.savefig(sub_sav_loc_eps, dpi=300, format='svg')

# Makes the gif
if save_movie == 1:
    print('Saving gif...')
    images = []
    for index in steps_sort:
        m = steps[index]
        sub_sav_loc = r'res/frame/movie/' + type[selected_type] + '_' + str(selected_run) + '_SUB_l' + str(
            remove_liquid) + '_m' + str(m) + '.png'
        images.append(imageio.imread(sub_sav_loc))
    gif_sav_loc = img_sav_loc = r'res/frame/gif/' + type[selected_type] + '_' + str(selected_run) + '_GIF_l' + str(remove_liquid) + '_m' + str(m) + '.gif'
    imageio.mimsave(gif_sav_loc, images, duration=0.5)



print('Done!')
    # graph_count += 1

    # dataframe.loc[files_png.index(j)] = [image_name] + [run_no] + [C] + [kT] + [mu0] + [m] + [fft_peak]

# print(str(1630-na_count) + ' of 1630 found peaks.')
# dataframe.to_csv(r'res/output.csv')

from scipy.optimize import curve_fit

# Function to calculate the power-law with constants a and b
def power_law(x, a, b):
    return a*np.power(x, b)



py.figure(4)
py.plot(np.arange(60,1520,20), height[1:], 'rx')
py.plot(np.arange(60,1520,20), power_law(np.arange(60,1520,20),1e9,0.25))
py.show()

# # Fit the dummy power-law data
# pars, cov = curve_fit(f=power_law, xdata=np.arange(60,1020,20), ydata=height[1:], p0=[0, 0], bounds=(-np.inf, np.inf))
# # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
# stdevs = np.sqrt(np.diag(cov))
# # Calculate the residuals
# # res = y_dummy - power_law(x_dummy, *pars)
#
# py.plot(np.arange(60,1020,20), power_law(np.arange(60,1020,20),pars))