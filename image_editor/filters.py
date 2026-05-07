import cv2
import numpy as np

# Grayscale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur
def gaussian_blur(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)

# Sharpen
def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

# Edge
def edge(img):
    return cv2.Canny(img, 100, 200)

# Brightness
def adjust_brightness(img, value):
    return cv2.convertScaleAbs(img, alpha=1, beta=value)

# Contrast
def adjust_contrast(img, value):
    return cv2.convertScaleAbs(img, alpha=value, beta=0)

# Flip
def flip(img):
    return cv2.flip(img, 1)

# Rotate
def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

# Noise
def add_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)








    