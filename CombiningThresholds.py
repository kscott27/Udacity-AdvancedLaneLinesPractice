import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from GradientHelpers import abs_sobel_thresh, mag_thresh, dir_threshold

# Read in an image
image = mpimg.imread('../images/signs_vehicles_xygrad.png')

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', thresh_min=30, thresh_max=100)
grady = abs_sobel_thresh(image, orient='y', thresh_min=30, thresh_max=100)
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

combined = np.zeros_like(image)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

plt.imshow(combined)
plt.savefig("../images/combined_thresh.jpg")