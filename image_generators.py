import matplotlib.image as mpimg
import cv2
from os import listdir
from os.path import isfile, join

def video_generator(path):
    vid = cv2.VideoCapture(path)

    if vid.isOpened() == False:
        print("Couldn't open the video")
        exit()

    while(vid.isOpened()):
        ret, image_bgr = vid.read()

        if not(ret):
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        yield image

def photo_generator(directory):

    test_files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]

    test_files.sort()

    print(test_files)

    for file in test_files:
        print("processing file ", file)
        image = mpimg.imread(file)
        yield image