'''
Functions used for Aiptasia Image Augmentation
1. Rotation
2. Reflection
3. Noise
'''

import cv2

def augmentation_v1(image):
  rotated_image = cv2.rotate(image, cv2.ROTATE_180)

  mirrored_image = cv2.flip(image, 1)

  rotated_then_mirrored_image = cv2.flip(rotated_image, 1)

  return rotated_image, mirrored_image, rotated_then_mirrored_image



