from os import listdir
from os.path import isfile, join
import cv2
from image_segmentation import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

import numpy as np


video_mode = True
photo_mode = False

def video_generator(path):
    vid = cv2.VideoCapture(video_file)

    if vid.isOpened() == False:
        print("Couldn't open the video")
        exit()

    while(vid.isOpened()):
        ret, image_bgr = vid.read()

        if not(ret):
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        yield image

def photo_generator(file_list):
    for file in test_files:
        print("processing file ", file)
        image = mpimg.imread(file)
        yield image


gen = None

if video_mode:
    video_file = "project_video.mp4"
    gen = video_generator(video_file)

elif photo_mode:

    test_images_path = "my_test_images"

    test_files = [join(test_images_path, f) for f in listdir(test_images_path) if isfile(join(test_images_path, f))]

    test_files.sort()

    print(test_files)

    gen = photo_generator(test_files)


for image in gen:

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    h = hls[:,:,0]

    l = hls[:,:,1]

    l = l / np.max(l) * 255.0

    s = hls[:,:,2]

    s = s / np.max(s) * 255.0

    yellow_s = np.zeros_like(s)
    yellow_s[(s >= int(0.4*255)) & (s <= 255)] = 1

    yellow_h = np.zeros_like(h)
    yellow_h[(h >= 20) & (h <= 50)] = 1

    print("done")

    if photo_mode:

        plt.subplot(331)
        plt.imshow(image)

        plt.subplot(332)
        plt.imshow(h, cmap=cm.gray, vmin=20, vmax=100) #hue

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
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image",  image_bgr)
        cv2.imshow("image_yellow",  yellow_h*255.0)
        cv2.waitKey(10)