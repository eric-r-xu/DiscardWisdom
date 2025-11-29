import cv2
import numpy as np
import os
import io
import pyautogui


def init_screenshot(image_path):
    shot = pyautogui.screenshot()  # PIL Image (RGB)
    shot.save(image_path)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Error: Could not read the saved screenshot at {image_path}")

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    return image, mask


def init_screenshot(image_path):
    shot = pyautogui.screenshot() 
    shot.save(image_path)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Error: Could not read the saved screenshot at {image_path}")

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    return image, mask


def find_num_locations_and_all_locations(template_image, image, mask, threshold = 0.8):
    num_locations = 0
    all_locations = []
    result = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED)
    matches = np.where(result >= threshold)
    for y, x in zip(*matches):
        end_y, end_x = y + template_image.shape[0], x + template_image.shape[1]
        if np.any(mask[y:end_y, x:end_x]):
            continue
        mask[y:end_y, x:end_x] = True
        all_locations.append([x, end_x, y, end_y])
        num_locations += 1

    return [num_locations, all_locations]



def return_first_match_location(template_image, image, threshold = 0.8):
    first_match_location = []
    result = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED)
    matches = np.where(result >= threshold)
    first_match_location = []
    for y, x in zip(*matches):
        end_y, end_x = y + template_image.shape[0], x + template_image.shape[1]
        first_match_location = [x, end_x, y, end_y]
        return first_match_location
    return first_match_location
