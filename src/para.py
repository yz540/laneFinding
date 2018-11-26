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
polyfit_search_margin = 100

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
