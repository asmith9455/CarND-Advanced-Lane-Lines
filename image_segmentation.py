import cv2

def to_hls(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    return hls

def to_yuv(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return yuv

# def select_yellow_1(image):
