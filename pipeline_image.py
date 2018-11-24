'''
Created on 22 Nov 2018

@author: yz540
'''
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import para

def camera_calibration_undistortion():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob('../camera_cal/calibration*.jpg')
    
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
#             cv2.imshow('img',img)
#             cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    # Camera calibration to get the camera matrix and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return mtx, dist

def produce_thresholded_combined_binary_image(img, gradient_threshold, s_threshold, gray_threshold):
    # get the s channel of the HLS space
    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]
    # get the grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # apply sobel on x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # Take the absolute value of the gradient
    abs_sobelx = np.absolute(sobelx)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_abs_x = np.uint8(255*abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_abs_x)
    sxbinary[(scaled_abs_x >= gradient_threshold[0]) & (scaled_abs_x <= gradient_threshold[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_threshold[0]) & (s_channel <= s_threshold[1])] = 1
    gray_binary = np.zeros_like(gray)
    gray_binary[(gray >= gray_threshold[0]) & (gray <= gray_threshold[1])] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(gray_binary ==1 ) & ((s_binary == 1) | (sxbinary == 1)) ] = 1
    return combined_binary

def get_perspective_transform_matrix(src, dst):
#     # Define src four corners
#     src = np.float32([[200, 720], [600, 445], [680, 445], [1080, 720]])
#     # Define dst four corners
#     dst = np.float32([[300, 720], [300, 0], [980, 0], [980, 720]])
    # Get the transform matrix
    return cv2.getPerspectiveTransform(src, dst),    cv2.getPerspectiveTransform(dst, src)

def get_top_down_view(binary_img):
    # Warp image to top-down view
    warped_img =  cv2.warpPerspective(binary_img, para.M, para.img_size, flags = cv2.INTER_LINEAR)
#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#     ax1.set_title('test image')
# #     cv2.polylines(img, np.array([src], dtype=np.int32), True, [0,255,0], 10)
#     ax1.imshow(binary_img)
#     
#     ax2.set_title('warped image')
# #     cv2.polylines(warped_img, np.array([dst], dtype=np.int32), True, [0,255,0], 10)
#     ax2.imshow(warped_img)
    return np.array(warped_img, dtype = np.uint8)  

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//3:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 8
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin 
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin 
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### Calc both polynomials using ploty, left_fit and right_fit ###
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        left_detected = True
    except TypeError:
        left_detected = False
        left_fitx = None
    
    try:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        right_detected = True
    except TypeError:
        right_detected = False
        right_fitx = None
    return left_fit, right_fit, left_fitx, right_fitx, ploty, left_detected, right_detected

def search_around_poly(binary_warped, pre_left_fit, pre_right_fit):
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = (  (nonzerox > (pre_left_fit[0]*(nonzeroy**2) + pre_left_fit[1]*nonzeroy + pre_left_fit[2] - para.polyfit_search_margin)) 
                      & (nonzerox < (pre_left_fit[0]*(nonzeroy**2) + pre_left_fit[1]*nonzeroy + pre_left_fit[2] + para.polyfit_search_margin)))
    
    right_lane_inds = ((nonzerox > (pre_right_fit[0]*(nonzeroy**2) + pre_right_fit[1]*nonzeroy + pre_right_fit[2] - para.polyfit_search_margin))
                      & (nonzerox < (pre_right_fit[0]*(nonzeroy**2) + pre_right_fit[1]*nonzeroy + pre_right_fit[2] + para.polyfit_search_margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty, left_detected, right_detected = \
    fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    return left_fit, right_fit, left_fitx, right_fitx, ploty, leftx, lefty, \
        rightx, righty, left_detected, right_detected
#     ## Visualization ##
#     # Create an image to draw on and an image to show the selection window
#     out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
#     window_img = np.zeros_like(out_img)
#     # Color in left and right line pixels
#     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
# 
#     # Generate a polygon to illustrate the search window area
#     # And recast the x and y points into usable format for cv2.fillPoly()
#     left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-para.polyfit_search_margin, ploty]))])
#     left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+para.polyfit_search_margin, 
#                               ploty])))])
#     left_line_pts = np.hstack((left_line_window1, left_line_window2))
#     right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-para.polyfit_search_margin, ploty]))])
#     right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+para.polyfit_search_margin, 
#                               ploty])))])
#     right_line_pts = np.hstack((right_line_window1, right_line_window2))
# 
#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
#     cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
#     result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    


# def measure_curvature_pixel(warped):
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
#     plt.savefig('../test_images/fittedPoly.png')
#     
#     # Define y-value where we want radius of curvature
#     # We'll choose the maximum y-value, corresponding to the bottom of the image
#     y_eval = np.max(ploty)
#     
#     # Calculation of R_curve (radius of curvature)
#     left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
#     right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
#     
#     return left_fit, right_fit, left_fitx, right_fitx, ploty, left_curverad, right_curverad

def measure_curvature_real(warped, leftx, lefty, rightx, righty, ploty):
    left_fit = np.polyfit(lefty * para.ym_per_pix, leftx * para.xm_per_pix, 2)
    right_fit = np.polyfit(righty * para.ym_per_pix, rightx * para.xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty) * para.ym_per_pix
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad

def draw_project_lines(image, warped, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, para.Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
#     plt.imshow(result)
    return result

# for fname in test_images:
#     print(fname)
#     img = cv2.imread(fname)
#     if M is None:
#         M, Minv = get_perspective_transform_matrix(img.shape[1::-1])
#         
#     # produce undistorted image after camera calibration
#     undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
#     cv2.imwrite(fname.replace('.jpg', '_undistorted.jpg'), undistorted_img)
#     # produce gradient and s channel thresholded binary image
#     binary_img = produce_thresholded_combined_binary_image(undistorted_img, gradient_threshold, s_threshold, gray_threshold)
#     cv2.imwrite(fname.replace('.jpg', '_thresholded.jpg'), binary_img*255)
#     # produce warped image after perspective transform
#     warped_img = get_top_down_view(binary_img)
#     cv2.imwrite(fname.replace('.jpg', '_warped.jpg'), warped_img*255)
#     # get pixel curvature
#     left_fitx, right_fitx, ploty, left_curverad, right_curverad = measure_curvature_pixel(warped_img, fname)
#     print('pixel curvature: ', left_curverad, right_curverad)
#     # get real curvature
#     left_curverad, right_curverad = measure_curvature_real(warped_img)
#     print('real curvature: ', left_curverad, right_curverad)
#     
#     protected_img = draw_project_lines(img, warped_img, left_fitx, right_fitx, ploty)
#     cv2.imwrite(fname.replace('.jpg', '_protected.jpg'), protected_img)