import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_generators import *

def get_yellow_and_white(image):

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

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

    return image_white_and_yellow_bin

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
        
            cv2.waitKey(3000)
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

            cv2.waitKey(3000)
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

    print("source: ", src)
    print("destination: ", dst)

    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(image, M, image.shape[::-1], flags=cv2.INTER_LINEAR)

    return warped

def draw_poly(image, poly_row_to_col):

    for row in range(image.shape[0]-1):
        col = int(np.polyval(poly_row_to_col, row))
        col2 = int(np.polyval(poly_row_to_col, row + 1))

        cv2.line(image, (col, row), (col2, row + 1), (0, 255, 0))



def extract_lanes(image):
    
    n_histo = 12 #number of histograms
    win_width = 40  #width of the sliding window
    req_frac = 0.5  #required fraction of the vertical slice that must be filled to be a max

    img_rows = image.shape[0]
    img_cols = image.shape[1]

    histo_height = int(img_rows / n_histo)

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

        max_col_left = np.argmax(histo[:histo_mp])
        max_col_right = np.argmax(histo[histo_mp:]) + histo_mp

        cnt_max_left = histo[max_col_left]
        cnt_max_right = histo[max_col_right]

        left_found = max_col_left != 0 and cnt_max_left >= int(req_frac * histo_height)
        right_found = max_col_right != histo_mp and cnt_max_right >= int(req_frac * histo_height)

        if left_found:
            max_col_left = max(win_width // 2, max_col_left)
            cv2.rectangle(image_with_lines, (max_col_left - win_width // 2, start_row), (max_col_left +  win_width // 2, end_row), (255,0,0))
            histo_mps_left.append([(start_row + end_row) // 2, max_col_left]) #row, column

        if right_found:
            max_col_right = min(img_cols - 1 - win_width // 2, max_col_right)
            cv2.rectangle(image_with_lines, (max_col_right - win_width // 2, start_row), (max_col_right +  win_width // 2, end_row), (0,0,255))
            histo_mps_right.append([(start_row + end_row) // 2, max_col_right]) #row, column
        
        

    for i in range(len(histo_mps_left) - 1):
        cv2.line(image_with_lines, tuple(histo_mps_left[i][::-1]), tuple(histo_mps_left[i+1][::-1]), (255,0,0))

    for i in range(len(histo_mps_right) - 1):
        cv2.line(image_with_lines, tuple(histo_mps_right[i][::-1]), tuple(histo_mps_right[i+1][::-1]), (0,0,255))

    histo_mps_left = np.array(histo_mps_left)
    histo_mps_right = np.array(histo_mps_right)

    # polynomial should map row to column
    left_poly = np.polyfit(histo_mps_left[:,0], histo_mps_left[:,1], 2)
    right_poly = np.polyfit(histo_mps_right[:,0], histo_mps_right[:,1], 2)

    draw_poly(image_with_lines, left_poly)
    draw_poly(image_with_lines, right_poly)

    return image_with_lines



        # print("histo: ", histo)
        
        # fig = plt.figure

        # plt.plot(histo)

        # plt.show()


        