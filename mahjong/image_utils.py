import cv2
import numpy as np
import io
from IPython.display import display
import PIL.Image as Image

def show_image(img, scale):
    resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    is_success, buffer = cv2.imencode(".png", resized_img)
    if not is_success:
        raise ValueError("Failed to convert the image to PNG buffer")
    io_buf = io.BytesIO(buffer)
    display(Image.open(io_buf))

def init_screenshot(image_path):
    width, height = 2778, 1284
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not read the target image.")
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask = np.zeros((height, width), dtype=bool)
    return resized_image, mask

def find_and_mark_locations(template_image, image, overlay, match_mask):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    threshold = 0.85
    locations = 0
    resized_template = original_template_gray
    result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
    matches = np.where(result >= threshold)
    for y, x in zip(*matches):
        end_y, end_x = y + resized_template.shape[0], x + resized_template.shape[1]
        if np.any(match_mask[y:end_y, x:end_x]):
            continue
        match_mask[y:end_y, x:end_x] = True
        cv2.rectangle(overlay, (x, y), (end_x, end_y), (0, 255, 255), -1)
        locations += 1
    return locations