import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from prev_poly import fit_poly, search_around_poly
from sliding_window import find_lane_pixels, fit_polynomial
from GradientHelpers import dir_threshold, mag_thresh
from process_image import process_image
from HelperFunctions import weighted_img, draw_lines, extrapolateLine

plt.ion()

image = mpimg.imread('../images/bridge_shadow.jpg')

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(15, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    h_thresh = (20,35)
    sy_thresh = (0,100)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(10, 255))

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_x = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Sobel y
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_y = np.uint8(255*abs_sobely/np.max(abs_sobely))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel_x)
    sxbinary[(scaled_sobel_x >= sx_thresh[0]) & (scaled_sobel_x <= sx_thresh[1])] = 1
    # Threshold y gradient
    sybinary = np.zeros_like(scaled_sobel_y)
    sybinary[(scaled_sobel_y >= sy_thresh[0]) & (scaled_sobel_y <= sy_thresh[1])] = 1
    
    # Threshold s-color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Threshold h-color channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    dir_binary = dir_threshold(image, sobel_kernel=9, thresh=(0.7, 1.3))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(dir_binary)
    # combined_binary[((s_binary == 1) & (h_binary == 1)) | (sxbinary == 1)] = 1
    # combined_binary[(( (sxbinary == 1)) | (dir_binary==1)) | ((s_binary==1) )] = 1
    # combined_binary[(h_binary==1)] = 1
    combined_binary[(((s_binary == 1) & (dir_binary==1)) | ((sxbinary == 1))) | ((h_binary==1))] = 1

    return combined_binary

combined_binary = pipeline(image)
combined_binary = np.uint8(255*combined_binary/np.max(combined_binary))


# This time we are defining a four sided polygon to mask
imshape = combined_binary.shape

imageHeight = imshape[0]
imageWidth = imshape[1]
img_size = (imageWidth, imageHeight)
upperLeftVertex = imageWidth/2 - imageWidth/34, imageHeight/1.8
upperRightVertex = imageWidth/2 + imageWidth/34, imageHeight/1.8
lowerRightVertex = imageWidth*0.925, imageHeight
lowerLeftVertex = imageWidth*0.075, imageHeight

# plt.imshow(combined_binary)
# plt.plot(720, 450, '.')
# plt.plot(1100, 700, '.')
# plt.plot(200, 700, '.')
# plt.plot(600, 450, '.')

src = np.float32([[720,450],
                  [1100,700],
                  [200,700],
                  [600,450]])

dst = np.float32([[1000,200],
                  [1000,700],
                  [200,700],
                  [200,200]])

# For source points I'm grabbing the outer four detected corners
# src = np.float32([upperRightVertex, lowerRightVertex, lowerLeftVertex, upperLeftVertex])
# # For destination points, I'm arbitrarily choosing some points to be
# # a nice fit for displaying our warped result 
# # again, not exact, but close enough for our purposes
# dst = np.float32([[700,400],[700,700],[400,700],[500,400]])
# Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)
# Warp the image using OpenCV warpPerspective()
warped = cv2.warpPerspective(combined_binary, M, img_size)

def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram

# Create histogram of image binary activations
# histogram = hist(warped)

# Visualize the resulting histogram
# plt.plot(histogram)



# Run image through the pipeline
# Note that in your project, you'll also want to feed in the previous fits
result = search_around_poly(warped)

# View your output
plt.imshow(result)

# processed_image = process_image(warped)

# color=[int(255),int(0),int(0)]
# cv2.line(combined_binary, lowerRightVertex, upperRightVertex, color)
# cv2.line(combined_binary, lowerLeftVertex, upperLeftVertex, color)

# Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()

# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=40)

# ax2.imshow(warped)
# ax2.set_title('Combined', fontsize=40)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.savefig('../images/pipeline_output.jpg')