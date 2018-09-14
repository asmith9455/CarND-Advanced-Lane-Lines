import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_generators import *

from collections import deque

import pprint

import math

pp = pprint.PrettyPrinter(indent=4)

def getEdges(single_channel_image):

    gray_f64 = single_channel_image.astype(np.float64)

    x_edges = cv2.Sobel(gray_f64, cv2.CV_64F, 1, 0, ksize=5)
    y_edges = cv2.Sobel(gray_f64, cv2.CV_64F, 0, 1, ksize=5)

    edges_mag = np.sqrt(np.square(x_edges) + np.square(y_edges))

    edges_dir = np.arctan2(np.absolute(y_edges), np.absolute(x_edges)) # (range 0 to pi / 2)

    return edges_mag, edges_dir

def get_yellow_and_white(image):

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    h = hls[:,:,0]

    l = hls[:,:,1]

    l = l

    s = hls[:,:,2]

    s = s / np.max(s) * 255.0

    yellow_s = np.zeros_like(s)
    yellow_s[(s >= int(0.4*255)) & (s <= 255)] = 1

    yellow_h = np.zeros_like(h)
    yellow_h[(h >= 20) & (h <= 50)] = 1

    white_l = np.zeros_like(l)
    white_l[(l >= 200)] = 1

    image_white_and_yellow_bin = cv2.bitwise_or(yellow_h, white_l)

    gray_f64 = gray.astype(np.float64)

    cv2.imshow("gray", gray)

    edges_mag, edges_dir = getEdges(gray)

    # edges_mag

    edges_mag = 255.0 * (edges_mag / np.max(edges_mag))
    edges_mag = edges_mag.astype(np.uint8)

    edges_dir_ang = edges_dir.copy()
    edges_dir = edges_dir / np.max(edges_dir) * 255.0
    edges_dir = edges_dir.astype(np.uint8)

    edges_mag_bin = np.zeros_like(edges_mag)
    edges_mag_bin[edges_mag > 10] = 255

    edges_dir_bin = np.zeros_like(edges_dir)
    edges_dir_bin[(edges_dir >= 1) & (edges_dir <= 120 )] = 255

    edges_overall_a = cv2.bitwise_and(edges_dir_bin, edges_mag_bin)

    edges_overall_b = cv2.bitwise_or(edges_overall_a, image_white_and_yellow_bin * 255)

    edges_overall_1 = cv2.erode(edges_overall_b, np.ones((3,3)),iterations=1)
    edges_overall_2 = cv2.dilate(edges_overall_1, np.ones((5,5)),iterations=5)
    edges_overall_3 = cv2.erode(edges_overall_2, np.ones((5,5)),iterations=4)

    cv2.imshow("white_and_yellow", image_white_and_yellow_bin * 255)

    # cv2.imshow("edges_mag", edges_mag)
    cv2.imshow("edges_mag_bin", edges_mag_bin)

    # cv2.imshow("edges_dir", edges_dir)
    cv2.imshow("edges_dir_bin", edges_dir_bin)

    cv2.imshow("edges_overall_a", edges_overall_a)
    cv2.imshow("edges_overall_b", edges_overall_b)
    cv2.imshow("edges_overall_1", edges_overall_1)
    cv2.imshow("edges_overall_2", edges_overall_2)
    cv2.imshow("edges_overall_3", edges_overall_3)

    # image_pp = cv2.bitwise_and(image_white_and_yellow_bin, x_edges)

    image_pp = image_white_and_yellow_bin
    image_pp = (edges_overall_3 / 255).astype(np.uint8)


    return image_pp

def get_lane_edges(image):

    x_edges = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)

    return x_edges

def calibrate_camera(directory, nx, ny, show_images=False):
    # directory is the path to a directory containing the calibration (checkerboard) images

    objpoints = []
    imgpoints = []
    gray_shape = None

    objp = np.zeros((9*6,3), dtype=np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    for image in photo_generator(directory):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape
        

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if not(ret):
            print("Couldn't find chessboard corners for a calibration image.")
            continue
        else:
            print("Found chessboard corners for a calibration image.")
        
        objpoints.append(objp)
        imgpoints.append(corners)

        if show_images and False:
            cv2.imshow("Original", image)

            cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
            cv2.imshow("Chessboard Corners", image)
        
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape[::-1], None, None)

    print("camera calibration success?: ", ret)

    if show_images:
        for image in photo_generator(directory):
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if not(ret):
                continue

            cv2.imshow("Original", image)
            cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
            cv2.imshow("Chessboard Corners", image)
            undist = cv2.undistort(image, mtx, dist, None, mtx)
            cv2.imshow("Undistorted", undist)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return mtx, dist

def perspective_tf_lane_lines(image):

    img_cols = image.shape[1]
    img_rows = image.shape[0]

    last_row = img_rows - 1
    last_col = img_cols - 1

    mid_col = int(img_cols * 0.5)
    mid_row = int(img_rows * 0.5)

    # src = \
    # [
    #     [mid_col - 50, mid_row],        # upper left
    #     [mid_col + 50, mid_row],        # upper right
    #     [mid_col - 200, last_row],      # lower left
    #     [mid_col + 200, last_row]       # lower right
    # ]

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

    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)

    Minv = cv2.getPerspectiveTransform(dst, src)

    if len(image.shape) == 3:
        warped_all = np.zeros_like(image)
        warped_all[:,:,0] = cv2.warpPerspective(image[:,:,0], M, image.shape[::-1][1:3], flags=cv2.INTER_LINEAR)
        warped_all[:,:,1] = cv2.warpPerspective(image[:,:,1], M, image.shape[::-1][1:3], flags=cv2.INTER_LINEAR)
        warped_all[:,:,2] = cv2.warpPerspective(image[:,:,2], M, image.shape[::-1][1:3], flags=cv2.INTER_LINEAR)
        return warped_all, M, Minv
    else:
        warped = cv2.warpPerspective(image, M, image.shape[::-1], flags=cv2.INTER_LINEAR)
        return warped, M, Minv 

    

def draw_poly(image, poly_row_to_col, width=1):

    for row in range(image.shape[0]-1):
        col = int(np.polyval(poly_row_to_col, row))
        col2 = int(np.polyval(poly_row_to_col, row + 1))

        cv2.line(image, (col, row), (col2, row + 1), (0, 255, 0), thickness=width)

def average_lane_buffer_order_2(buffer):

    buff_sum = np.array([0,0,0], dtype=np.float32)

    for lane in buffer:
        for i in range(3):
            buff_sum[i] = float(buff_sum[i]) + float(lane[i])
    
    L = float(len(buffer))

    buff_avg = [ float(buff_sum[0]) / L, float(buff_sum[1]) / L, float(buff_sum[2]) / L]

    pp.pprint(buffer)
    pp.pprint(buff_sum)
    pp.pprint(buff_avg)

    return buff_avg

class LaneExtractor(object):

    def __init__(self):
        self.n_histo = 30 #number of histograms
        self.win_width = 80  #width of the sliding window
        self.req_frac = 0.9  #required fraction of the vertical slice that must be filled to be a max

        self.average_len = 2

        self.left_lane_buffer = deque(maxlen=self.average_len)
        self.right_lane_buffer = deque(maxlen=self.average_len)

    def process_image(self, image):
        
        image = image.copy()

        self.shape = image.shape

        img_rows = image.shape[0]
        img_cols = image.shape[1]

        histo_height = int(img_rows / self.n_histo)

        print('histo_height: ', histo_height)

        histo_mps_left = []
        histo_mps_right = []

        image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for start_row in range(img_rows - 1, -1, -histo_height):
            print ("computing histo")
            end_row = start_row - histo_height + 1
            print("start_row: ", start_row)
            print("end_row: ", end_row)
            print("image_shape: ", image.shape)
            #print("image: ", image[start_row:end_row, :])
            histo = np.sum(image[end_row:start_row, :], axis=0)

            histo_mp = histo.shape[0] // 2

            # max_col_left = np.argmax(histo[:histo_mp])
            # max_col_right = np.argmax(histo[histo_mp:]) + histo_mp

            # cnt_max_left = histo[max_col_left]
            # cnt_max_right = histo[max_col_right]

            # left_found = None

            # if len(histo_mps_left) == 0:
            #     left_found = max_col_left != 0 and cnt_max_left >= int(self.req_frac * histo_height)
            # else:
            #     left_found = max_col_left != 0 and cnt_max_left >= int(self.req_frac * histo_height) and abs(max_col_left - histo_mps_left[-1][1]) < self.win_width // 2

            # right_found = max_col_right != histo_mp and cnt_max_right >= int(self.req_frac * histo_height)

            # print("left found? : ", left_found)

            # if left_found:
                
            #     max_col_left = max(self.win_width // 2, max_col_left)
            #     cv2.rectangle(image_with_lines, (max_col_left - self.win_width // 2, start_row), (max_col_left +  self.win_width // 2, end_row), (255,0,0))
            #     histo_mps_left.append([(start_row + end_row) // 2, max_col_left]) #row, column

                
            # if right_found:

            #     max_col_right = min(img_cols - 1 - self.win_width // 2, max_col_right)
            #     cv2.rectangle(image_with_lines, (max_col_right - self.win_width // 2, start_row), (max_col_right +  self.win_width // 2, end_row), (0,0,255))
            #     

            

            req_cont_frac = 0.9
            req_cont_height = req_cont_frac * histo_height
            req_num_reps = 25
            max_jump_between_frames = 50

            # -------------------------------------------------------------
            # match on the right side--------------------------------------
            # -------------------------------------------------------------

            right_ctr = 0
            right_reps_ctr = 0
            right_found = False
            right_match_col = 0

            for col in histo[histo_mp:]:
                if col > req_cont_height:
                    right_reps_ctr = right_reps_ctr + 1
                else:
                    right_reps_ctr = 0
                    if right_found:
                        break
                
                if right_reps_ctr >= req_num_reps:
                    right_found = True
                    right_match_col = right_ctr - right_reps_ctr // 2
                
                right_ctr = right_ctr + 1

            right_match_row = (start_row + end_row) // 2
            right_match_col = right_match_col + histo_mp

            ret, right_lane = self.right_lane()

            if ret and right_found:
                right_found = right_found and max_jump_between_frames >= abs( int(np.polyval(right_lane, right_match_row)) - right_match_col )

            if right_found:
                histo_mps_right.append([right_match_row, right_match_col]) #row, column 
                cv2.rectangle(image_with_lines, (right_match_col - self.win_width // 2, start_row), (right_match_col +  self.win_width // 2, end_row), (0,0,255))

            # -------------------------------------------------------------
            # -------------------------------------------------------------
            # -------------------------------------------------------------

            # -------------------------------------------------------------
            # match on the left side--------------------------------------
            # -------------------------------------------------------------

            left_ctr = 0
            left_reps_ctr = 0
            left_found = False
            left_match_col = 0

            for col in histo[:histo_mp][::-1]:
                if col > req_cont_height:
                    left_reps_ctr = left_reps_ctr + 1
                else:
                    left_reps_ctr = 0
                    if left_found:
                        break
                
                if left_reps_ctr >= req_num_reps:
                    left_found = True
                    left_match_col = histo_mp - (left_ctr - left_reps_ctr // 2)
                
                left_ctr = left_ctr + 1

            # must be within the max_jump_between_frames of the previous polynomial in order to be 'found'

            left_match_row = (start_row + end_row) // 2

            ret, left_lane = self.left_lane()

            if ret and left_found:
                left_found = left_found and max_jump_between_frames >= abs( int(np.polyval(left_lane, left_match_row)) - left_match_col )

            if left_found:
                left_match_col = left_match_col
                histo_mps_left.append([left_match_row, left_match_col]) #row, column 
                cv2.rectangle(image_with_lines, (left_match_col - self.win_width // 2, start_row), (left_match_col +  self.win_width // 2, end_row), (0,0,255))

            # -------------------------------------------------------------
            # -------------------------------------------------------------
            # -------------------------------------------------------------

                
        if len(histo_mps_left) > 2:
        
            for i in range(len(histo_mps_left) - 1):
                cv2.line(image_with_lines, tuple(histo_mps_left[i][::-1]), tuple(histo_mps_left[i+1][::-1]), (255,0,0))

            histo_mps_left = np.array(histo_mps_left)
            left_poly = np.polyfit(histo_mps_left[:,0], histo_mps_left[:,1], 2) # polynomial maps row to column
            draw_poly(image_with_lines, left_poly)
            self.left_lane_buffer.append(left_poly)

        if len(histo_mps_right) > 2:

            for i in range(len(histo_mps_right) - 1):
                cv2.line(image_with_lines, tuple(histo_mps_right[i][::-1]), tuple(histo_mps_right[i+1][::-1]), (0,0,255))

            histo_mps_right = np.array(histo_mps_right)
            right_poly = np.polyfit(histo_mps_right[:,0], histo_mps_right[:,1], 2)
            draw_poly(image_with_lines, right_poly)
            self.right_lane_buffer.append(right_poly)


        return image_with_lines
        
    def left_lane(self):
        if len(self.left_lane_buffer) == 0:
            return False, None
        else:
            return True, average_lane_buffer_order_2(self.left_lane_buffer)
        
    def right_lane(self):
        if len(self.right_lane_buffer) == 0:
            return False, None
        else:
            return True, average_lane_buffer_order_2(self.right_lane_buffer)

    def left_curvature(self):
        ret, lane = self.left_lane()

        px_2_m_col2y = 3.7 / 690.0 # since the lane width is 3.7 m
        px_2_m_row2x = 40.0 / 720.0  # use the length of a dashed lane line in the transformed image

        if not(ret):
            return False, 0.0
        
        A = lane[0] * px_2_m_col2y / px_2_m_row2x**2    # adjust polynomials such that we move into real world dimensions rather than pixel dimensions
        B = lane[1] * px_2_m_col2y / px_2_m_row2x       
        C = lane[2] * px_2_m_col2y
        
        row = (self.shape[0] - 1) * px_2_m_col2y

        curv = (1.0 + (2.0*A*row + B )**2.0 )**(1.5) / abs(2.0 * A)

        return True, curv

    def right_curvature(self):
        ret, lane = self.right_lane()

        px_2_m_col2y = 3.7 / 690.0 # since the lane width is 3.7 m
        px_2_m_row2x = 40.0 / 720.0  # use the length of a dashed lane line in the transformed image

        if not(ret):
            return False, 0.0
        
        A = lane[0] * px_2_m_col2y / px_2_m_row2x**2    # adjust polynomials such that we move into real world dimensions rather than pixel dimensions
        B = lane[1] * px_2_m_col2y / px_2_m_row2x
        C = lane[2] * px_2_m_col2y
        
        row = (self.shape[0] - 1) * px_2_m_col2y

        curv = (1.0 + (2.0*A*row + B )**2.0 )**(1.5) / abs(2.0 * A)

        return True, curv
    
    def get_mid_column(self, row, center_col):

        # 690 px is 3.7 m

        px_2_m_col2y = 3.7 / 690.0 # since the lane width is 3.7 m
        px_2_m_row2x = 40.0 / 720.0  # use the length of a dashed lane line in the transformed image

        left_col = int(np.polyval(self.left_lane()[1], row))
        right_col = int(np.polyval(self.right_lane()[1], row))

        mid_col = (left_col + right_col) // 2

        dist_2_center = (mid_col - center_col) * px_2_m_col2y

        width = abs(right_col - left_col) * px_2_m_col2y

        return dist_2_center, width

        