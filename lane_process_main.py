# opencv
import cv2

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

# my code
from lane_line_img_processing import *
from image_generators import *

# numpy
import numpy as np


video_mode = True
photo_mode = False

gen = None

if video_mode:
    gen = video_generator("project_video.mp4")

elif photo_mode:
    gen = photo_generator("my_test_images")

# perform camera calibration

mtx, dist = calibrate_camera("camera_cal", 9, 6, False)

lane_extractor = LaneExtractor()

for image in gen:

    image_undist = cv2.undistort(image, mtx, dist, None, mtx)

    image_white_and_yellow_bin = get_yellow_and_white(image_undist)

    ptrans_image, M, Minv = perspective_tf_lane_lines(image_white_and_yellow_bin)

    # lane_image = extract_lanes(ptrans_image)

    print("done")

    if photo_mode:

        plt.subplot(331)
        plt.imshow(image)
        plt.title("original image")

        plt.subplot(332)
        #plt.imshow(h, cmap=cm.gray, vmin=20, vmax=100) #hue
        plt.imshow(image_undist)
        plt.title("undistorted image")

        plt.subplot(333)
        plt.imshow(image_white_and_yellow_bin)
        plt.title("white and yellow binary")

        plt.subplot(334)
        plt.imshow(l, cmap='gray') #lightness
        plt.title("lightness")
        plt.subplot(335)
        plt.imshow(h, cmap='gray') #hue
        plt.subplot(336)
        plt.title("hue")
        plt.imshow(s, cmap='gray') #saturation
        plt.title("saturation")
        plt.subplot(338)
        plt.imshow(yellow_h, cmap=cm.gray, vmin=0, vmax=1)
        plt.subplot(339)
        # plt.imshow(yellow_s, cmap=cm.gray, vmin=0, vmax=1)
        plt.imshow(yellow_s, cmap='gray')

        # cv2.imshow("yellow select", yellow_select)

        # cv2.waitKey(0)

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        plt.show()


    elif video_mode:
        image_undist_bgr = cv2.cvtColor(image_undist, cv2.COLOR_RGB2BGR)

        debug_image = lane_extractor.process_image(ptrans_image)

        left_lane_exists, left_lane = lane_extractor.left_lane()

        right_lane_exists, right_lane = lane_extractor.right_lane()

        lane_lines_p_frame = np.zeros_like(image_undist_bgr)

        if left_lane_exists:
            draw_poly(lane_lines_p_frame, left_lane, width=20)

        if right_lane_exists:
            draw_poly(lane_lines_p_frame, right_lane, width=20)

        lanes_orig_frame = \
            cv2.warpPerspective(lane_lines_p_frame, Minv, lane_lines_p_frame.shape[1::-1], cv2.INTER_LINEAR)

        lanes_orig_frame = cv2.addWeighted(image_undist_bgr, 1.0, lanes_orig_frame, 1.0, 0.0)

        cv2.imshow("image", image_undist_bgr)
        cv2.imshow("image_white_and_yellow_bin", image_white_and_yellow_bin*255.0)
        cv2.imshow("perspective transform", ptrans_image*255.0)
        cv2.imshow("debug_image", debug_image*255.0)
        cv2.imshow("lanes_orig_frame", lanes_orig_frame)
        # cv2.imshow("drawn_lines_img_unperspective", drawn_lane_lines_unperspective)
        # cv2.imshow("lanes image", lane_image)
        
        cv2.waitKey(10)