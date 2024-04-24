import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
from IPython.display import display
import PIL.Image as Image
import io
import os


def show_image(img, scale):
    """Resize and display an image in Jupyter Notebook."""
    resized_img = cv2.resize(
        img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    is_success, buffer = cv2.imencode(".png", resized_img)
    if not is_success:
        raise ValueError("Failed to convert the image to PNG buffer")
    io_buf = io.BytesIO(buffer)
    display(Image.open(io_buf))


def find_and_print_locations(template_path, image_path):
    """Find, print, and display each location of the template image in the target image with different scaling."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the target image.")
        return

    original_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if original_template is None:
        print("Error: Could not read the template image.")
        return

    show_image(original_template, 0.5)  # Show the original template scaled down by 50%

    scales = [
        2.8,
        2.6,
        2.4,
        2.2,
        2.0,
        1.8,
        1.6,
        1.4,
        1.2,
        1,
        0.8,
        0.6,
        0.4
    ]  # Different scales to try

    locations = 0
    match_mask = np.zeros(image.shape[:2], dtype=bool)

    for scale in scales:
        resized_template = cv2.resize(
            original_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        result = cv2.matchTemplate(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            resized_template,
            cv2.TM_CCOEFF_NORMED,
        )
        threshold = 0.84
        matches = np.where(result >= threshold)

        for y, x in zip(*matches):
            end_y, end_x = y + resized_template.shape[0], x + resized_template.shape[1]

            if np.any(match_mask[y:end_y, x:end_x]):
                continue  # Skip overlapping matches

            match_mask[y:end_y, x:end_x] = True

            highlight_image = image.copy()
            cv2.rectangle(highlight_image, (x, y), (end_x, end_y), (0, 255, 255), -1)
            cv2.addWeighted(highlight_image, 0.7, image, 0.3, 0, highlight_image)
            # show_image(highlight_image, 0.3)  # Display the match scaled down by 50%
            locations += 1
        if locations > 0:
            print(f"Scale {scale*100}% found {locations} matches")

    if locations > 4:
        raise ValueError("ERROR: more than 4 matches found!")
    return locations


# Paths and execution code
screenshot_path = "/path/to/your/screenshot.jpeg"
template_dir = "/path/to/your/templates/"
template_files = []

# Use os.walk to find all files in directory and subdirectories
for root, dirs, files in os.walk(template_dir):
    for file in files:
        if file.endswith(".PNG") or file.endswith(".png"):
            # Append the full path of the file
            template_files.append(os.path.join(root, file))

tiles_found = defaultdict(int)

prev = ""
for template_path in template_files:
    if prev != os.path.basename(os.path.dirname(template_path)):
        try:
            print(f"total {prev} tiles found = {tiles_found[prev]}")
        except:
            pass
        print(
            f"---------------------{os.path.basename(os.path.dirname(template_path))}---------------------"
        )
        prev = os.path.basename(os.path.dirname(template_path))
    found_tiles = find_and_print_locations(template_path, screenshot_path)

    if found_tiles > 0:
        print(f"Found tiles={found_tiles}")
    tiles_found[os.path.basename(os.path.dirname(template_path))] += found_tiles

try:
    print(f"total {prev} tiles found = {tiles_found[prev]}")
except:
    pass

for k,v in tiles_found.items():
    print(f"{k} tiles = {v}")
print("Made it to the bitter end!")