import cv2
import numpy as np
import io
from IPython.display import display
import PIL.Image as Image
import os
from collections import defaultdict
import sys
from datetime import datetime


def show_image(img, scale):
    resized_img = cv2.resize(
        img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    is_success, buffer = cv2.imencode(".png", resized_img)
    if not is_success:
        raise ValueError("Failed to convert the image to PNG buffer")
    io_buf = io.BytesIO(buffer)
    display(Image.open(io_buf))


def init_screenshot(image_path):
    width, height = 1280, 800
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not read the target image.")
    resized_image = cv2.resize(
        image, (width, height), interpolation=cv2.INTER_AREA)
    mask = np.zeros((height, width), dtype=bool)
    return resized_image, mask


def find_and_mark_locations(template_image, image, overlay, match_mask):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    threshold = 0.85
    locations = 0
    resized_template = original_template_gray
    result = cv2.matchTemplate(
        image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
    matches = np.where(result >= threshold)
    for y, x in zip(*matches):
        end_y, end_x = y + \
            resized_template.shape[0], x + resized_template.shape[1]
        if np.any(match_mask[y:end_y, x:end_x]):
            continue
        match_mask[y:end_y, x:end_x] = True
        cv2.rectangle(overlay, (x, y), (end_x, end_y), (0, 255, 255), -1)
        locations += 1
    return locations


def load_images_from_folder(folder, prefix):
    """Load images from the specified folder with a specific prefix."""
    images = []
    filenames = []
    for filename in os.listdir(folder):
        filename = filename.lower()
        if filename.startswith(prefix) and filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames


def load_templates_from_folder(folder):
    """Load template images from the specified folder."""
    templates = []
    filenames = []
    for filename in os.listdir(folder):
        filename = filename.lower()
        if filename.endswith(".png"):
            template_path = os.path.join(folder, filename)
            template = cv2.imread(template_path)
            if template is not None:
                templates.append(template)
                filenames.append(filename)
    return templates, filenames


def perform_template_matching(image: str, template: str, threshold: float) -> float:
    """Perform template matching on a given image with a given template.
       Filter results based on a threshold of match (0-1).
       Return whether a match is found.
    """
    template_h, template_w, _ = template.shape

    result_channels = []
    for c in range(template.shape[2]):  # Iterate over each color channel
        img_channel = image[:, :, c]
        template_channel = template[:, :, c]
        res = cv2.matchTemplate(
            img_channel, template_channel, cv2.TM_CCOEFF_NORMED)
        result_channels.append(res)

    # Combine the results by averaging
    combined_result = np.mean(result_channels, axis=0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(combined_result)

    if max_val >= threshold:
        return max_val
    else:
        return 0


def check_for_template_match(screenshot_path, template_path, threshold, coord_range=None):
    screenshot = cv2.imread(screenshot_path)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if screenshot is None or template is None:
        raise FileNotFoundError(
            f"Could not read {screenshot_path} or {template_path}")

    if len(coord_range) > 0:
        x_min = coord_range.get("x_min", 0)
        x_max = coord_range.get("x_max", screenshot.shape[1])

        y_min = coord_range.get("y_min", 0)
        y_max = coord_range.get("y_max", screenshot.shape[0])

        screenshot = screenshot[y_min:y_max, x_min:x_max]

    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    min_x, max_x, min_y, max_y = None, None, None, None
    loc = np.where(res >= threshold)

    match_found = int(len(loc[0]) > 0)

    if match_found == 1:
        min_x, max_x, min_y, max_y = min(loc[1]), max(
            loc[1]), min(loc[0]), max(loc[0])

    elif match_found == 0:
        while threshold > 0.93:
            threshold -= 0.01
            _loc = np.where(res >= threshold)
            if len(_loc[0]) > 0:
                min_x, max_x, min_y, max_y = min(_loc[1]), max(
                    _loc[1]), min(_loc[0]), max(_loc[0])

    return bool(match_found), min_x, max_x, min_y, max_y, round(threshold, 2)
