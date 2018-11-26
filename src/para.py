'''
Created on 23 Nov 2018

@author: yz540
'''
img_size = None
M = None
Minv = None
mtx = None # camera calibration matrix
dist = None # camera calibration distortion coefficient

# gradient threshold
gradient_threshold = [30, 100]
# color channel threshold
s_threshold = [120, 255]
gray_threshold = [100, 255]

# Choose the width of the margin around the previous polynomial to search
polyfit_search_margin = 80

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# record the left and right lines data in the 5 last frames
SMOOTH_WINDOW = 5

# HYPERPARAMETERS for sliding windows in pixel searching
# Choose the number of sliding windows
nwindows = 8
# Set the width of the windows +/- margin
margin = 80
# Set minimum number of pixels found to recenter window
minpix = 80