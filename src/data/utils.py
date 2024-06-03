import os
import cv2
import numpy as np


def save_annotated_image(image, label, save_path):
    image = reverse_transform(image)
    label = reverse_transform(label)
    image_with_lanes = cv2.addWeighted(image, 1, label, 1, 0)
    cv2.imwrite(save_path, image_with_lanes)


def reverse_transform(img):
    img = np.transpose(img, (1, 2, 0)).dot(255).astype(np.uint8)
    if img.shape[2] == 1:
        red_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        red_mask[:, :, 2] = img[:, :, 0]
        img = red_mask
    return img


def makedir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
