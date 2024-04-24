import cv2
import numpy as np
import os
from collections import defaultdict


def show_image(img, scale):
    from IPython.display import display
    import PIL.Image as Image
    import io

    resized_img = cv2.resize(
        img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    is_success, buffer = cv2.imencode(".png", resized_img)
    if not is_success:
        raise ValueError("Failed to convert the image to PNG buffer")
    io_buf = io.BytesIO(buffer)
    display(Image.open(io_buf))


def init_screenshot(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not read the target image.")
    return image, np.zeros(image.shape[:2], dtype=bool)


def find_and_print_locations(template_image, image, match_mask):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    threshold = 0.9
    locations = 0

    resized_template = original_template_gray
    result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
    matches = np.where(result >= threshold)

    for y, x in zip(*matches):
        end_y, end_x = y + resized_template.shape[0], x + resized_template.shape[1]
        if np.any(match_mask[y:end_y, x:end_x]):
            continue
        match_mask[y:end_y, x:end_x] = True
        highlight_image = image.copy()
        cv2.rectangle(highlight_image, (x, y), (end_x, end_y), (0, 255, 255), -1)
        cv2.addWeighted(highlight_image, 0.7, image, 0.3, 0, highlight_image)
        show_image(highlight_image, 0.7)
        locations += 1

    return locations


screenshot_path = "/Users/ericrxu/Jupyter/screenshot_test.jpeg"
template_dir = "/Users/ericrxu/Jupyter/hkmj_tile_templates/"

template_files = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(template_dir)
    for file in files
    if file.endswith((".PNG", ".png"))
]

image, match_mask = init_screenshot(screenshot_path)
templates = {file: cv2.imread(file) for file in template_files}

tiles_found = defaultdict(int)

for template_path, template_image in templates.items():
    _template_id = template_path.split("/")[-3:]
    _player, _type, _tile = (
        _template_id[0],
        _template_id[1],
        _template_id[2].replace(".png", ""),
    )
    if template_image is None:
        print(f"Error: Could not read the template image at {template_path}")
        continue
    # print(f"Processing template: {template_path.split('/')[-3:]}")
    found_tiles = find_and_print_locations(template_image, image, match_mask)
    if found_tiles > 0:
        print(f"Found {found_tiles} {_tile} {_type} tile(s) for {_player}")
        tiles_found[_tile] += found_tiles



print(
    f"\n------------------------------------------'OVERALL'------------------------------------------\n"
)
for k, v in tiles_found.items():
    print(f"\n{v} {k} tiles found\n")

print("Made it to the bitter end!")