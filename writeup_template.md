**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration2_src.jpg "Distorted"
[image2]: ./output_images/calibration2_dst.jpg "Undistorted"
[image3]: ./test_images/test1.jpg "Road img"
[image4]: ./output_images/test1_undistorted.png "Undistorted road img"
[image5]: ./output_images/test1_thresholded.png "Binary img"
[image6]: ./output_images/test1_warped.png "Warp img"
[image7]: ./output_images/test1_fittedPoly.png "Fit Visual"
[image8]: ./output_images/test1_projected.png "Output img"
[video1]: ./project_video.mp4 "Video"
[video2]: ./project_video_output.mp4 "Result video"
[video3]: ./challenge_video.mp4 "Chanllenge video"
[video4]: ./challenge_video_output.mp4 "Challenge result video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 13 - 46 in Python file `src/pipeline_image.py`.  The tutorial of camera calibration in openCV can be found [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result. For instance, the calibration2_src is the distorted image with corners detected. 

![calibration2.jpg][image1]
![calibration2 undistorted][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Road image][image3]
In the camera calibration step, I obtained the camera matrix and distortion coefficients. By calling the method cv2.undistort(), I got the undistorted image like this one:
![Undistorted road image][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

At first I used a combination of s channel of HSV color space, grayscale threshold and sobel gradient thresholds to generate a binary image (thresholding steps at lines 66 through 94 in method produce_thresholded_combined_binary_image() in`src/pipeline_image.py` where lines 88 through 94 was commented because it didn't work well for challenge video.). The combined binary is:
    combined_binary[(gray_binary ==1 ) & ((s_binary == 1) | (sxbinary == 1)) ] = 1

Although it works even with shadows in the project video with these three thresholds, it failed to detect the yellow lane lines in the challenge video. I changed it to detect yellow in the left half and white in the right half. 

Here's an example of my output for this step.
![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_top_down_view()`, which includes two parts:
* define source (`src`) and destination (`dst`) points once for all the transformation of images under testing. These points are defined in lines 20 through 32 in the file "src/test_lanelinesinImage.py".
* call the method get_perspective_transform_matrix(src, dst) to get the transform matrix from src to dst and from dst to src in lines 97 through 99 in "src/pipeline_image.py".
```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warped binary image][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I took a histogram of the bottom half of the image, then found the peak of the left and right halves of the histogram and used them as the base of the two lane lines on left and right side. Then I iterated sliding windows to detect nonzero pixels. The corresponding code can be found in lines 114 through 182 in method find_lane_pixels() in `src/pipeline_image.py`. 
With the nonzero x and y values, I fit my lane lines with a 2nd order polynomial in lines 184 through 207 in method fit_poly() in `src/pipeline_image.py`. The resulting fitted lines are like this:

![polyfitted image][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 277 through 289 in my code in method measure_curvature_real() in `src/pipeline_image.py`. The formula used is in this [page](https://www.intmath.com/applications-differentiation/8-radius-curvature.php)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 291 through 309 in method draw_project_lines() in `src/pipeline_image.py`.  Here is an example of my result on a test image:

![Output image][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [Project video result][video2]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I changed the color threshold method to detect yellow in the left half of the image, white in the right half of the image combined with the gradient threshold. It worked better than the one combined s-channel, grayscale and gradient, but in some images useful pixels are still not extracted correctly.

I implemented the tricks suggested in the project like recorded the last 5 frames to smooth the resulting lines. When the detected lines are reasonable, I search only the pixels within neighbouring area of the previous fit polynomial lines to avoid blind sliding window search. But when the detected lines are problematic, these method do not help much to produce good results.

My solution is highly dependent on the color and gradient threshold precision. It is not robust and often fails when there is a big piece of shadow, or bright light that confuses the yellow color detection. It only works when the car is on the left side of the road. When there is a sharp turn, it cannot detect the pixels correctly. It might help if I can tell the difference of the road image to use different method to detect.
