'''
Created on 22 Nov 2018

@author: yz540
'''
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from pipeline_image import *
import para
import Line
from collections import deque

def sanity_check(left_line, right_line):
# check similar curvature
    similar_curvature = np.absolute(left_line.radius_of_curvature - right_line.radius_of_curvature) / max(left_line.radius_of_curvature, right_line.radius_of_curvature) < 0.3  
# check two lines are separated by approximately the right distance horizontally
    right_distance = np.absolute(left_line.line_base_pos + right_line.line_base_pos - 3.7) < 1
# check roughly parallel, horizon distance between two lines are similar from top to down, use standard deviation of the distances to check
    horizon_distance = [np.absolute(left_line.bestx[i] - right_line.bestx[i]) for i in range(para.img_size[1])]
    standard_deviation = np.std(horizon_distance)
    nearly_parallel = standard_deviation / np.mean(horizon_distance) < 0.3
    return similar_curvature and right_distance and nearly_parallel
  
def process_image(img):
    from builtins import str
    global passed_sanity_check, left_lines, right_lines, recent_right_xfitted, recent_left_xfitted
    # When read the first frame, assign the constant values to variables that will be 
    # used everywhere but only need to assign once in the project
    if para.img_size is None:
        # image size
        para.img_size = img.shape[1::-1]
        
        # src four corners for perspective transform
        src = np.float32(
        [[para.img_size[0] / 2 - 60, para.img_size[1] / 2 + 100],
        [para.img_size[0] / 6 - 10, para.img_size[1]],
        [para.img_size[0] * 5 / 6 + 60, para.img_size[1]],
        [para.img_size[0] / 2 + 60, para.img_size[1] / 2 + 100]])
        
        # dst four corners for perspective transform
        dst = np.float32(
        [[(para.img_size[0] / 4), 0],
        [(para.img_size[0] / 4), para.img_size[1]],
        [(para.img_size[0] * 3 / 4), para.img_size[1]],
        [(para.img_size[0] * 3 / 4), 0]])
        
        # perspective transform matrix from src to dst and the inverse
        para.M, para.Minv = get_perspective_transform_matrix(src, dst)

    # The pipeline to process each image: 
    # 1> produce undistorted image after camera calibration
    undistorted_img = cv2.undistort(img, para.mtx, para.dist, None, para.mtx)
    # 2> produce gradient and s channel thresholded binary image
    binary_img = produce_thresholded_combined_binary_image(undistorted_img, para.gradient_threshold, para.s_threshold, para.gray_threshold)
    # 3> produce warped image after perspective transform
    warped_img = get_top_down_view(binary_img)
    # 4> fit the left and right polynomial lines from the pixels in the warped binary image from the above step
    if passed_sanity_check and len(left_lines) > 0:
        # otherwise, search within the neighbouring area of the previous polynomial lines.
        # get related info of the fitted lines
        left_fit, right_fit, left_fitx, right_fitx, \
        ploty, leftx, lefty, rightx, righty, left_detected, right_detected = \
        search_around_poly(warped_img, left_lines[-1].best_fit, right_lines[-1].best_fit)
    else:    
        # find lane pixels using sliding windows
        leftx, lefty, rightx, righty, warpage = find_lane_pixels(warped_img)
        # get the second degree polynomial fit
        left_fit, right_fit, left_fitx, right_fitx, ploty, left_detected, right_detected = \
        fit_poly(para.img_size, leftx, lefty, rightx, righty)
    
    if left_detected and right_detected:
        # get real curvature
        left_curverad, right_curverad = measure_curvature_real(warped_img, leftx, lefty, rightx, righty, ploty)
    #     print('real curvature: ', left_curverad, right_curverad)
    
        # create two new lines and add to the end of the deque    
        left_line = Line.Line()
        right_line = Line.Line()
        
        # put data into the two lines
        left_line.allx = leftx
        left_line.ally = lefty
        left_line.current_fit = left_fit
        left_line.radius_of_curvature = left_curverad
        left_line.detected = left_detected
        
        if len(left_lines) >= 1:
            left_line.diffs = np.absolute(left_lines[-1].current_fit - left_line.current_fit)
            right_line.diffs = np.absolute(right_lines[-1].current_fit - right_line.current_fit)
        recent_left_xfitted.append(left_fitx)
        left_line.recent_xfitted = recent_left_xfitted
        
        left_line.bestx = np.average(left_line.recent_xfitted, 0)
        left_line.line_base_pos = (para.img_size[0]/2 - left_line.bestx[-1]) * para.xm_per_pix
        
        right_line.allx = rightx
        right_line.ally = righty
        right_line.current_fit = right_fit
        right_line.radius_of_curvature = right_curverad
        right_line.detected = right_detected
        recent_right_xfitted.append(right_fitx)
        right_line.recent_xfitted = recent_right_xfitted
        right_line.bestx = np.average(right_line.recent_xfitted, 0)
        right_line.line_base_pos = (right_line.bestx[-1] - para.img_size[0]/2) * para.xm_per_pix
    
        distance_from_centre = (left_line.line_base_pos + right_line.line_base_pos - 3.7)/2
        if np.absolute(distance_from_centre) > 0.5:
            print("wrong centre offset!!!!!!!!!!!!!!")
        # perform sanity check for each frame
        passed_sanity_check = sanity_check(left_line, right_line)
        if passed_sanity_check:
            print("passed sanity check!!!!!!!")
                # the best fit is the average of the fit in the last 5 frames
            left_lines.append(left_line)
            right_lines.append(right_line)        
            left_line.best_fit = np.sum([l.current_fit for l in left_lines], 0) / len(left_lines)
            right_line.best_fit = np.sum([r.current_fit for r in right_lines], 0) / len(right_lines)
        elif len(left_lines) > 0:
            left_lines.popleft()
            right_lines.popleft()
            
        if len(left_lines) > 0:
            left_bestx = left_lines[-1].bestx
            right_bestx = right_lines[-1].bestx
        else:
            left_bestx = left_line.bestx
            right_bestx = right_line.bestx
    
        projected_img = draw_project_lines(undistorted_img, warped_img, left_bestx, right_bestx, ploty)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (0,0,0)
        lineType = 2
        
        cv2.putText(projected_img, "Curvature radius:" , (10, 50), font, fontScale, fontColor, lineType)
        cv2.putText(projected_img, "Left: " + str("%.2f" % left_line.radius_of_curvature) + "m Right: " + str("%.2f" % right_line.radius_of_curvature) + "m" , (10, 100), font, fontScale, fontColor, lineType)
        cv2.putText(projected_img, "Distance from centre: " , (10, 150), font, fontScale, fontColor, lineType)
        cv2.putText(projected_img, str("%.2f" % distance_from_centre) + "m" , (10, 200), font, fontScale, fontColor, lineType)
        #     cv2.imwrite(fname.replace('.jpg', '_protected.jpg'), projected_img)
        
        # return the final output (image where lines are drawn on lanes)
        return projected_img
    else:
        return img




# Only calibrate the camera once at the beginning
if para.mtx == None:
    para.mtx, para.dist = camera_calibration_undistortion()

# store the recent fitted x 
passed_sanity_check = False
recent_left_xfitted = deque(maxlen = para.SMOOTH_WINDOW)
recent_right_xfitted = deque(maxlen = para.SMOOTH_WINDOW)

# record lines info of the last SMOOTH_WINDOW frames
left_lines = deque(maxlen = para.SMOOTH_WINDOW)
right_lines = deque(maxlen = para.SMOOTH_WINDOW)

# Process each image in the input video
test_video_output = '../challenge_video_output.mp4'
clip1 = VideoFileClip("../challenge_video.mp4")
test_video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
test_video_clip.write_videofile(test_video_output, audio=False)