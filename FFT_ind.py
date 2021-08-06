# Find the radially averaged 1D FFT power spectrum of one image

from scipy import fftpack
# import pyfits
import numpy as np
import pylab as py
import numpy.matlib as ml
import matplotlib.pyplot as plt
from skimage.filters import window
from scipy import signal


def radial_profile2(data, center=None, binning=2):
    y, x = np.indices((data.shape))  # first determine radii of all pixels

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # radius of the image.
    r_max = np.max(r)
    bin_no = np.rint(r_max/binning).astype(int)

    ring_brightness, radius = np.histogram(r, weights=data, bins=bin_no)

    return ring_brightness


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def normalise(array):  # Set an image numpy array's values to be between 0 and 1
    norm_array = (array-np.min(array))\
                         / (np.max(array)-np.min(array))
    return norm_array

# image = r'res1/C8_1th_spin_0000_3_0.png'
# image = r'res1/Si_t10_spin_05mgmL_0001_2_0.png'


# image = 'Paper/SiPatternedAu1_750020.png'
# image_name = '71066_outfile_C6_kT45_MR1_mu0285_muf15_vus75_sig01_L1024_m2200_IC.png'
image_name = '41756_outfile_C6_kT325_MR1_mu028_muf15_vus75_sig01_L1024_m3000_IC.png'
image_loc = r'Data/' + image_name
image = plt.imread(image_loc)
image = rgb2gray(image)

# image = np.random.uniform(0, 1, [1024, 1024])

# Generate an image that is a sinwave propagated over every row of frequency f
# f = 140
# image = normalise(ml.repmat(np.sin(f*2*np.pi*np.linspace(0, 1, 1024)), 1024, 1)) # Make a sinosoid horizontal row and repeat it down the array

# Window the image using Hann
image_hann = image * window('hann', image.shape)


# Take the fourier transform of the image.
F1 = fftpack.fft2(image)


# Now shift the quadrants around so that low spatial frequencies are in
# the center of the 2D fourier transformed image.
F2 = fftpack.fftshift(F1)

# Calculate a 2D power spectrum by squaring
psd2D = np.abs(F2)**2

# Calculate the radially averaged 1D power spectrum

psd1D = radial_profile2(psd2D)

# #Generate frequency axis
# n = len(psd1D)
# fr = np.linspace(0,1,n)

F1_hann = fftpack.fft2(image_hann)
F2_hann = fftpack.fftshift(F1_hann)
psd2D_hann = np.abs(F2_hann)**2
psd1D_hann = radial_profile2(psd2D_hann)

# Now plot up both
py.figure(1)
py.clf()
py.imshow(image, cmap=py.cm.Greys)

py.figure(2)
py.clf()
psd2D[psd2D > np.mean(psd2D)] = 0
py.imshow(psd2D, cmap=py.cm.hsv)
# sav_loc_2dfft = r'res/2dfft/2D_FFT' + image_name
# py.savefig(sav_loc_2dfft, format='png', dpi=300)

py.figure(3)
py.clf()
x, y = image.shape
horz = np.linspace(0, np.sqrt((x/2)**2+(y/2)**2), 362)  # Use Pythagoras to work out the furthest radius on the image
py.semilogy(horz, psd1D)



# py.semilogy(np.linspace(0, 512, 362),psd1D_hann)
py.xlabel('Spatial Frequency/Hz')
py.ylabel('Power Spectrum')

peaks, properties = signal.find_peaks(np.log(psd1D), prominence=1)
py.plot(horz[peaks], psd1D[peaks], 'X', label='Unsmoothed peaks')
print(horz[peaks])  # Print the peak spatial frequency to the console

py.show()