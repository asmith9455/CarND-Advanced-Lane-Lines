from os import listdir
from os.path import isfile, join
import cv2
from image_segmentation import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

import numpy as np

test_images_path = "my_test_images"

test_files = [join(test_images_path, f) for f in listdir(test_images_path) if isfile(join(test_images_path, f))]

test_files.sort()

print(test_files)

for file in test_files:
    print("processing file ", file)
    
    image = mpimg.imread(file)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    h = hls[:,:,0]

    l = hls[:,:,1]

    l = l / np.max(l) * 255.0

    s = hls[:,:,2]

    s = s / np.max(s) * 255.0

    yellow_s = np.zeros_like(s)
    yellow_s[(s >= 200) & (s <= 255)] = 1

    yellow_h = np.zeros_like(h)
    yellow_h[(h >= 40) & (h <= 80)] = 1

    print("done")

    plt.subplot(331)
    plt.imshow(image)
    plt.subplot(334)
    plt.imshow(l, cmap='gray')
    plt.subplot(335)
    plt.imshow(h, cmap='gray')
    plt.subplot(336)
    plt.imshow(s, cmap='gray')
    plt.subplot(338)
    plt.imshow(yellow_h, cmap=cm.gray, vmin=0, vmax=1)
    plt.subplot(339)
    # plt.imshow(yellow_s, cmap=cm.gray, vmin=0, vmax=1)
    plt.imshow(yellow_s, cmap='gray')

    # cv2.imshow("yellow select", yellow_select)

    # cv2.waitKey(0)

    plt.show()

    break