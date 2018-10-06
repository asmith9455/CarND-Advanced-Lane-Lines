## Advanced Lane Lines Project Writeup

---

**Goals**

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lane_line_img_processing.py, lines 118 through 186.

The first step in the calibration is collecting object points. These are 3D points in the world. The world frame is defined such that the origin is the upper left of the checkerboard pattern  in the images, the x axis is along one side of the checkerboard and the y axis is along the other. The checkerboard is the plane z = 0. Therefore the image points are the same for all calibration images. The 3D points just need to be scaled properly - the actual size of the squares was not taken into account for this calibration since not supplied (assumed 1m by 1m squares).

Then, 2D image points are calculated for each image using the `cv2.findChessboardCorners` function. These are pixel positions in the calibration images.

The 2D and 3D points from each image are all appending to object point and image point arrays, then fed into the `cv2.calibrateCamera` function. Note that in some of the images, not all of the chessboard corners are visible. In this case, the image and object points were not appended to their respective arrays. 

I applied the resultant distortion correction matrix to the calibratoin images using `cv2.undistort` function and obtained the following chart as examples. I chose these examples because the pattern clearly becomes straight after being curved.

Original             |  Chessboard Corners | Undistorted |
|:-------------------------:|:-------------------------:|:------:|
|![Image 2 Original](report_imgs/calibration_2_original.png)  |  ![Image 2 Chessboard](report_imgs/calibration_2_chessboard_corners.png) | ![Image 2 Chessboard](report_imgs/calibration_2_undistorted.png)|
|![Image 3 Original](report_imgs/calibration_3_original.png)  |  ![Image 3 Chessboard](report_imgs/calibration_3_chessboard.png) | ![Image 3 Chessboard](report_imgs/calibration_3_undistorted.png)|

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

| Original Image | Distortion-Corrected Image |
|:---:|:---:|
| ![Original Image](report_imgs/pipe_imgs/image_original.png) | ![Undistorted Image](report_imgs/pipe_imgs/image_undistorted.png) |

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 40 through 120 in `lane_process_main.py`). Note that I performed the image processing after performing the perspective transform on the captured image.

First, I calculated the gradient direction and magnitude, and thresholded each to produce two binary images where I expect the lane lines to be. These are shown in the table below.

| Gradient Magnitude Thresholded | Gradient Direction Thresholded |
|:---:|:---:|
|![Gradient Magnitude Thresholded](report_imgs/pipe_imgs/edges_mag_bin.png)|![Gradient Direction Thresholded](report_imgs/pipe_imgs/edges_dir_bin.png)|

Then, I calculated the bitwise_and of these two images to produce an edge based estimate of the lane line positions. I also applies colour filters to identify the white and yellow portions of the image. See the result of each of these thresholds in the table below.


| Lane Lines from Edge Information | Lane Lines from Colour Information |
|:---:|:---:|
|![Lane Lines from Edge Information](report_imgs/pipe_imgs/edges_overall_a.png) | ![Lane Lines from Colour Information](report_imgs/pipe_imgs/white_and_yellow.png)|

Then, I bitwise_or'd the two of these to produce an overall estimate of the lane lines position. I used the OR operation in this step because I hoped that in challenging conditions, the information from each approach could be combined to produce an estimate of the lane line position. I manually verified that each method independently had a low rate of false positives.

![Lane Lines Binary Image](report_imgs/pipe_imgs/edges_overall_b.png)

Finally, I performed
- an erosion to remove noise
- a dilation to make the lane lines whole
- another erosion to return the lane lines to close to their original size

![Lane Lines from Edge Information](report_imgs/pipe_imgs/edges_overall_1.png)
![Lane Lines from Edge Information](report_imgs/pipe_imgs/edges_overall_2.png)
![Lane Lines from Edge Information](report_imgs/pipe_imgs/edges_overall_3.png)

This last image is the binary image output that I used to pull lane line information from.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is contained in the function perspective_tf_lane_lines in lane_line_img_processing.py. Lines 188 to 232 of this file define the source and destination points. This code is also shown below. Then, I used the function `cv2.getPerspectiveTransform` to generate the perspective transformation matrix. Finally, I used the `cv2.warpPerspective` function to warp the image (named `image`) that was passed into the function.

```python

img_cols = image.shape[1]
img_rows = image.shape[0]

last_row = img_rows - 1
last_col = img_cols - 1

mid_col = int(img_cols * 0.5)
mid_row = int(img_rows * 0.5)

src = \
[
    [mid_col - 100, int(img_rows*0.63)],        # upper left
    [mid_col + 100, int(img_rows*0.63)],        # upper right
    [mid_col - 850, last_row],                  # lower left
    [mid_col + 850, last_row]                   # lower right
]

dst = \
[
    [0, 0],                                     # upper left
    [last_col, 0],                              # upper right
    [0, last_row],                              # lower left
    [last_col, last_row]                        # lower right
]

```

This resulted in the following source and destination points:


[[0, 0], [1279, 0], [0, 719], [1279, 719]]
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 540, 453      | 0, 0        | 
| 740, 453      | 1279, 0      |
| -210, 719     | 0, 719      |
| 1490, 719      | 1279, 719        |

Note that the source points on the bottom row actually extend outside the bottom of the image.

I verified that my perspective transform was working as expected by checking whether or not the lane lines are vertical on a straight section of road (for example, see the table below).

| Source        | Destination (Perspective Transform)   | 
|:-------------:|:-------------:| 
| ![Source Image](report_imgs/pipe_imgs/image_straight.png) | ![Destination Image](report_imgs/pipe_imgs/ptrans_straight.png) | 

Here is another example of the perspective transform, this time on a curved road.

| Source        | Destination (Perspective Transform)   | 
|:-------------:|:-------------:| 
| ![Source Image](report_imgs/pipe_imgs/image_undistorted.png) | ![Destination Image](report_imgs/pipe_imgs/ptrans_image_undist.png) | 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To produce an estimate of the lane line positions within the resultant binary image, I used a technique similar to the one presented in the lane finding lessons. This involves scanning through horizontal sections of the image (accessed using numpy code like `binary_image[start_row:end_row, :]`). I then computed a histogram of filled pixels by column for each of the horizontal sections. I then search for a particular quality of the resultant signal that seemed to robustly identify the position of the lane line in that particular section. Note that I always started scanning from the center column (histo_mp in the code) towards the outside of the image. The associated code is in lane_line_img_processing.py, lines 300 through 442 (part of the LaneExtractor class). 

My 'lane line found' criteria was at least 25 repeated histogram bins that were at least 90% filled (vertically). In addition, I required that any new lane line detection be at most 50 pixels away from the last estimate of the lane line position. 

With boxes around the seemingly most likely lane line pixels identified, I was able to take the centroids of the boxes and use them in a polynomial regression to produce polynomial estimates of each lane line. The below image shows the boxes that represent the estimated positions of the left and right lane lines in red and blue, respectively. Also, the resultant polynomials are drawn in green on the image (in the perspective transformed frame). 

![Lane Line Identification](report_imgs/pipe_imgs/debug_image.png)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Lane line curvature is calculated in lines lane_line_processing.py lines 462 through 504. Position of the vehicle relative to the center of the lane is accomplished in lane_line_processing.py lines 506 through 524. To calculate the curvatures, I used estimates of factors that allowed for conversion from pixel values to real world dimensions. The factors I ended up using are (3.7 / 690.0) [m/px] (in the y or lateral direction) and (40.0 / 720.0) [m/px]  (in the x or longitudinal direction).

Once I had calculated these factors, I adjusted the polynomials (which were detected in pixel space in the perspective transformed image) based on the process:

- col = A (row) ^ 2 + B (row) + C
- y = col * col2y
- y = col2y * ( A (row) ^ 2 + B (row) + C )
- x = row * row2x
- row = x / row2x
- y = col2y * ( A (x / row2x) ^ 2 + B (x / row2x) + C )
- y = col2y * A / (row2x ^ 2) * x ^ 2 + col2y * B / (row2x) * x + col2y * C
- x -> (row2x) * row

These are the equations I have used in the code. Since the estimated curvature radii of the various turns in the test video are approximately 1 km (as was mentioned in the project introduction as a test of the general correctness of the calculation), I believe that the calculations yield approximately accurate results.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 69 through 79 in my code in `lane_process_main.py` in the function `get_images()`.  Here is an example of my result on a test image:

![Results Plotted on the Road](report_imgs/pipe_imgs/lanes_orig_frame.png)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major challenge in this project was enabling reasonably robust identification of lane line pixels (from which the lane line polynomials were derived). In particular, generation of the binary image based on colour and edge segmentation was challenging. I really dislike manually tuning things like thresholds on the various features (such as intensity of certain colour channels or edge direction and magnitude), as the approach is not feasible for anything more than a few videos. Even then, it seems to me unlikely that the optimal result will be obtained by manual tuning. Rather, some sort of machine learning algorithm seems more efficient - however it requires the generation of a robust (labelled) dataset. The data itself is easy to collect, but labelling it is both time consuming and tedious. My pipeline will fail whenever it reaches a condition where the colour segmentation and edge segmentation fail to pick out the lane lines from the persepective transformed image. To make it more robust, I could either manually tune it in more scenarios or setup some sort of automated machine learning approach as I previously mentioned.
