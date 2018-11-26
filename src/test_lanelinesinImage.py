'''
Created on 26 Nov 2018

@author: yz540
'''
from pipeline_image import *
import para
test_images = glob.glob('../test_images/test[0-9].jpg')
# Only calibrate the camera once at the beginning
if para.mtx == None:
    para.mtx, para.dist = camera_calibration_undistortion()

for fname in test_images:
    print(fname)
#     img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    img = mpimg.imread(fname)
    if para.img_size == None:
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
    # 4> find lane pixels using sliding windows
    leftx, lefty, rightx, righty, warpage = find_lane_pixels(warped_img)
    # 5> get the second degree polynomial fit
    left_fit, right_fit, left_fitx, right_fitx, ploty, left_detected, right_detected = \
    fit_poly(para.img_size, leftx, lefty, rightx, righty)

    mpimg.imsave(fname.replace('.jpg', '_undistorted.png'), undistorted_img)
    mpimg.imsave(fname.replace('.jpg', '_thresholded.png'), binary_img*255)
    mpimg.imsave(fname.replace('.jpg', '_warped.png'), warped_img*255)
     
    protected_img = draw_project_lines(img, warped_img, left_fitx, right_fitx, ploty)
    mpimg.imsave(fname.replace('.jpg', '_projected.png'), protected_img)