'''
Functions used for Aiptasia Image Augmentation
1. Rotation
2. Reflection
3. Noise
'''

import cv2
import numpy as np

def image_rotate_180(image):
    return cv2.rotate(image, cv2.ROTATE_180)

def image_flip(image):
    return cv2.flip(image, 1)

def label_rotate_180():
    pass

def label_flip():
    pass


