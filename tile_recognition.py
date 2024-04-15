from IPython.display import display
import PIL.Image as Image
import io
import cv2
import numpy as np

def show_image(img):
    """Convert an image matrix into a displayable image in Jupyter Notebook."""
    is_success, buffer = cv2.imencode(".png", img)
    if not is_success:
        raise ValueError("Failed to convert the image to PNG buffer")
    io_buf = io.BytesIO(buffer)
    display(Image.open(io_buf))

def find_and_print_locations(template_path, image_path):
    """
    This function finds, prints, and displays each location of the template image in the target image with highlighted matches,
    considering rotations and scaling of the template.
    Highlights are shown in yellow with 30% opacity.
    """
    image = cv2.imread(image_path)  # Read image in color
    if image is None:
        print("Error: Could not read the target image.")
        return

    original_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if original_template is None:
        print("Error: Could not read the template image.")
        return

    angles = [0, 90, -90]  # Degrees of rotation
    scales = [1, 0.5]  # 50% original size

    for angle in angles:
        M = cv2.getRotationMatrix2D((original_template.shape[1] / 2, original_template.shape[0] / 2), angle, 1)
        rotated_template = cv2.warpAffine(original_template, M, (original_template.shape[1], original_template.shape[0]))

        for scale in scales:

            resized_template = cv2.resize(rotated_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if resized_template.shape[0] > image.shape[0] or resized_template.shape[1] > image.shape[1]:
                continue  # Skip if resized template is larger than the image

            result = cv2.matchTemplate(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), resized_template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            matches = np.where(result >= threshold)
            match_mask = np.zeros(image.shape[:2], dtype=bool)  # Create a mask for the whole image

            for y, x in zip(*matches):
                end_y, end_x = y + resized_template.shape[0], x + resized_template.shape[1]
                
                # Check if the area for this match is already covered
                if np.any(match_mask[y:end_y, x:end_x]):
                    continue  # Skip this match if any part of it is already covered

                # Mark this area as matched in the mask
                match_mask[y:end_y, x:end_x] = True

                print(f"Match found at: (y, x) = ({y}, {x}), angle = {angle}°, scale = {int(scale*100)}%")
                highlight_image = image.copy()
                cv2.rectangle(highlight_image, (x, y), (end_x, end_y), (0, 255, 255), -1)  # Yellow fill
                cv2.addWeighted(highlight_image, 0.7, image, 0.3, 0, highlight_image)  # 30% opacity
                show_image(highlight_image)  # Display each highlighted image

# Example usage
template_path = "./tiles/5d.jpg"
image_path = "screenshot.jpg"
find_and_print_locations(template_path, image_path)
