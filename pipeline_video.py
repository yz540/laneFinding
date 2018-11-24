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

previous_left_fit = np.array([0, 0, 0])
previous_right_fit =  np.array([0, 0, 0])
left_lines = deque(maxlen=5)
right_lines = deque(maxlen=5)

def sanity_check():
    if len(left_lines) == 0:
        return False
    else:
        left_line = left_lines[-1]
        right_line = right_lines[-1]
    # check similar curvature
        similar_curvature = np.absolute(left_line.radius_of_curvature - right_line.radius_of_curvature) / right_line.radius_of_curvature < 0.2  
    # check two lines are separated by approximately the right distance horizontally
        right_distance = np.absolute(left_line.line_base_pos + right_line.line_base_pos - 3.7) < 1
    # check roughly parallel, horizon distance between two lines are similar from top to down, use standard deviation of the distances to check
        horizon_distance = [np.absolute(left_line.allx[i] - right_line.allx[i]) for i in range(para.img_size[0])]
        standard_deviation = np.std(horizon_distance)
        nearly_parallel = standard_deviation < 0.5 
      
        return similar_curvature and right_distance and nearly_parallel
  
def process_image(img):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    if para.img_size is None:
        para.img_size = img.shape[1::-1]
        # src corners for perspective transform
        src = np.float32(
        [[(para.img_size[0] / 2) - 60, para.img_size[1] / 2 + 95],
        [((para.img_size[0] / 6) - 10), para.img_size[1]],
        [(para.img_size[0] * 5 / 6) + 60, para.img_size[1]],
        [(para.img_size[0] / 2 + 60), para.img_size[1] / 2 + 95]])
        
        # dst corners for perspective transform
        dst = np.float32(
        [[(para.img_size[0] / 4), 0],
        [(para.img_size[0] / 4), para.img_size[1]],
        [(para.img_size[0] * 3 / 4), para.img_size[1]],
        [(para.img_size[0] * 3 / 4), 0]])
        
        para.M, para.Minv = get_perspective_transform_matrix(src, dst)

        
    # produce undistorted image after camera calibration
    undistorted_img = cv2.undistort(img, para.mtx, para.dist, None, para.mtx)
    # produce gradient and s channel thresholded binary image
    binary_img = produce_thresholded_combined_binary_image(undistorted_img, para.gradient_threshold, para.s_threshold, para.gray_threshold)
    # produce warped image after perspective transform
    warped_img = get_top_down_view(binary_img)
    if not sanity_check():
        # find lane pixels using sliding windows
        leftx, lefty, rightx, righty, warpage = find_lane_pixels(warped_img)
        # get the second degree polynomial fit
        previous_left_fit = np.polyfit(lefty, leftx, 2)
        previous_right_fit = np.polyfit(righty, rightx, 2)

    left_fit, right_fit, left_fitx, right_fitx, \
    ploty, leftx, lefty, rightx, righty, left_detected, right_detected = \
    search_around_poly(warped_img, previous_left_fit, previous_right_fit)
    
#     print('pixel curvature: ', left_curverad, right_curverad)
    # get real curvature
    left_curverad, right_curverad = measure_curvature_real(warped_img, leftx, lefty, rightx, righty, ploty)
#     print('real curvature: ', left_curverad, right_curverad)
    
    left_line = Line.Line()
    right_line = Line.Line()
    
    '''TODO put data into the two lines'''
    left_line.allx = leftx
    left_line.ally = lefty
    left_line.current_fit = left_fit
    left_line.radius_of_curvature = left_curverad
    left_line.detected = left_detected
    if len(left_lines) >= 1:
        left_line.diffs = left_lines[-1].current_fit - left_line.current_fit
        right_line.diffs = right_lines[-1].current_fit - right_line.current_fit

    left_line.recent_xfitted.append(left_fitx)
    left_line.bestx = np.average(left_line.recent_xfitted, 0)
    left_line.line_base_pos = (para.img_size[1]/2 - left_line.bestx[-1]) * para.xm_per_pix
    
    right_line.allx = rightx
    right_line.ally = righty
    right_line.current_fit = right_fit
    right_line.radius_of_curvature = right_curverad
    right_line.detected = right_detected
    right_line.recent_xfitted.append(right_fitx)
    right_line.bestx = np.average(right_line.recent_xfitted, 0)
    right_line.line_base_pos = (right_line.bestx[-1] - para.img_size[1]/2) * para.xm_per_pix
    
    left_lines.append(left_line)
    right_lines.append(right_line)
   
    left_line.best_fit = np.sum([l.current_fit for l in left_lines]) / 5
    right_line.best_fit = np.sum([r.current_fit for r in right_lines]) / 5

    projected_img = draw_project_lines(img, warped_img, left_line.bestx, right_line.bestx, ploty)
    print(left_line.detected, right_line.detected)
    
#     cv2.imwrite(fname.replace('.jpg', '_protected.jpg'), projected_img)
    previous_left_fit = left_fit
    previous_right_fit = right_fit
    
    return projected_img

# Only calibrate once
if para.mtx == None:
    para.mtx, para.dist = camera_calibration_undistortion()


test_video_output = '../project_video_output.mp4'
clip1 = VideoFileClip("../project_video.mp4")
test_video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
test_video_clip.write_videofile(test_video_output, audio=False)