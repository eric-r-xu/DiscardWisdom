
import os
import sys
import time
from datetime import datetime
from io import StringIO
import random
import cv2
import numpy as np
import pyautogui
from pync import Notifier

from image_utils import check_for_template_match


class Config:
    NO_MOTION_SEC_THRESHOLD = 0.05
    HEARTBEAT_SEC = 60  # time between '...'
    NO_MOTION_CLICK_THRESHOLD = 0.08
    SAMPLING_RATE_FPS = 100
    MINUTES_BEFORE_STOPPING = 500
    COORDINATES = {
        "D1": (216, 541),
        "D2": (279, 541),
        "D3": (342, 541),
        "D4": (406, 541),
        "D5": (469, 541),
        "D6": (532, 541),
        "D7": (596, 541),
        "D8": (659, 541),
        "D9": (722, 541),
        "D10": (786, 541),
        "D11": (849, 541),
        "D12": (912, 541),
        "D13": (975, 541),
        "wall_tile": (1065, 539),
        "next_game": (893, 626),
        "accept": (1073, 449),
        "reject": (1163, 444),
        "exit_ad": (1205, 191),
        "draw_game": (618, 450),
        "unpause": (544, 592),
        "exit_ad2": (1202, 197),
        "exit_ad3": (1243, 141),
        "exit_ad4": (668, 604),
        "x": (1118, 200),
        "close_ad": (1198, 147)
    }
    # Add search ranges for templates for search efficiency and precision
    TEMPLATE_RANGES = {
        "NextGame": {"x_min": 1720, "y_min": 1220},
        "Pong": {"x_min": 1990, "y_min": 820},
        "Chow": {"x_min": 1930, "y_min": 820},
        "DetermineWinner": {"x_min": 0, "y_min": 400, "x_max": 1250 ,"y_max":800},
        "DetermineSelfPick": {"x_min": 1650, "y_min": 400 ,"y_max":1200},
        "DetermineWinnerFan": {"x_min": 1650, "y_min": 400 ,"y_max":1200},
    }


class Utils:
    @staticmethod
    def shuffle_dict_with_fixed_keys(d, fixed_keys):
        # shuffles dictionary keys with fixed keys at the beginning
        if not all(key in d for key in fixed_keys):
            raise KeyError("One or more fixed keys are not in the dictionary.")

        fixed_items = [(key, d[key]) for key in fixed_keys]
        remaining_items = [
            (key, value) for key, value in d.items() if key not in fixed_keys
        ]

        random.shuffle(fixed_items)
        random.shuffle(remaining_items)

        shuffled_dict = {key: value for key, value in fixed_items}
        shuffled_dict.update({key: value for key, value in remaining_items})

        return shuffled_dict


class ScreenshotCapturer:
    @staticmethod
    def capture():
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)


class MotionDetector:
    def __init__(self, threshold=50):
        self.threshold = threshold

    def detect(self, frame1, frame2):
        # omit top and bottom black border regions as well as ads in motion detection
        frame1[0:200, :, :] = 0
        frame2[0:200, :, :] = 0
        frame1[1190:, 0:1150, :] = 0
        frame2[1190:, 0:1150, :] = 0
        
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        _, diff_thresh = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return len(contours) > 0

    @staticmethod
    def show_image(image, title="Image"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()


class ClickMotionDetector(MotionDetector):
    def detect_after_click(self, location):
        frame1 = ScreenshotCapturer.capture()
        pyautogui.moveTo(location)
        msg = f"Moved mouse to {location}"
        # print('#########################################')
        # print('    ', msg)
        # print('#########################################')
        pyautogui.click()
        time.sleep(Config.NO_MOTION_CLICK_THRESHOLD)
        frame2 = ScreenshotCapturer.capture()
        return self.detect(frame1, frame2)


class MotionDetectionScript:
    def __init__(self):
        self.motion_detector = MotionDetector()
        self.click_motion_detector = ClickMotionDetector()
        self.start_time = time.time()
        self.no_motion_start_time = None
        self.screenshot_dir = self.create_screenshot_directory()

    def create_screenshot_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        dir_path = os.path.join("auto_screenshots", timestamp)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def save_screenshot(self, frame, filename):
        full_path = os.path.join(self.screenshot_dir, filename)
        frame = ScreenshotCapturer.capture()
        cv2.imwrite(full_path, frame)
        msg = f"Screenshot saved @ {full_path}"
        print("\n\n------------------------------------------------")
        print(msg)
        Notifier.notify(msg)
        return full_path

    def handle_no_motion(self, frame):
        coordinates_to_click_first = [
            "exit_ad4", "exit_ad3", "exit_ad", "exit_ad2", "x", "close_ad"]
        filename = f"X{int(time.time())}.png"
        full_path = self.save_screenshot(frame, filename)

        templates = {
            "ChowSelection": (
                "wall_tile",
                "D3",
                "D4",
                "D5",
                "D6",
                "D7",
                "D8",
                "D9",
                "D10",
                "D11",
                "D12",
                "D13",
            ),
            "KongPong": ("accept", "reject"),
            "PongChow": ("accept", "reject"),
            "NextGame": ("next_game",),
            "Pong": ("accept", "reject"),
            "Chow": ("accept", "reject"),
            "Ad": ("x", "close_ad"),
            "Kong": ("accept", "reject"),
            "DetermineSelfPick": ("next_game",),
            "DetermineWinner": ("next_game",),
            "DetermineWinnerFan": ("next_game",),
            "Draw": ("draw_game",),
        }
        found_any_match = 0
        template_addl = ''
        for template, coords in templates.items():
            template_range = Config.TEMPLATE_RANGES.get(template, {})
            template_dir = os.path.join(os.getcwd(), "templates", template)

            for template_file in sorted(os.listdir(template_dir)):
                template_dir.split('/')[-1]
                msg = f"{template_dir.split('/')[-1]}"



                if template_file.lower().endswith(".png"):
                    template_path = os.path.join(template_dir, template_file)
                    template_name_msg = f"{template_file.split('/')[-1]}"
                    match_found, min_x, max_x, min_y, max_y, threshold = (
                        check_for_template_match(
                            full_path, template_path, 0.97, template_range
                        )
                    )
                    if match_found == True:
                        found_any_match = 1
                        if template in ('DetermineWinner','DetermineSelfPick','DetermineWinnerFan'):
                            template_addl += '_'
                            template_addl += template_name_msg
                        msg = f"{template} ({template_name_msg}) found at (x range, y range) = ({min_x} to {max_x}, {min_y} to {max_y}) at {threshold}!"
                        print('#########################################')
                        print('        ', msg)
                        print('#########################################')
                        Notifier.notify(msg)
                        coordinates_to_click_first = coords
                        
                    else:
                        if min_x is not None:
                            msg = f"{template} found at (x range, y range) = ({min_x} to {max_x}, {min_y} to {max_y}) at {threshold}!"

                            print('            ', msg)


        if found_any_match == 1:
            self.save_screenshot(frame, f"{template}{template_addl}_" + filename)
            msg = f"removing {full_path}"
            print('        ', msg)
            os.remove(full_path)
            

        shuffled_dict = Utils.shuffle_dict_with_fixed_keys(
            Config.COORDINATES, coordinates_to_click_first
        )

        for tile_name, coordinates in shuffled_dict.items():
            if self.click_motion_detector.detect_after_click(coordinates):
                msg = f"Motion detected after clicking {tile_name} at {coordinates}"
                Notifier.notify(msg)
                print('#########################################')
                print('    ', msg)
                print('#########################################')
                break
        time.sleep(1 / Config.SAMPLING_RATE_FPS)
        self.no_motion_start_time = None

    def run(self):
        frame1 = ScreenshotCapturer.capture()
        time.sleep(1 / Config.SAMPLING_RATE_FPS)
        last_heartbeat = time.time()

        while True:
            frame2 = ScreenshotCapturer.capture()
            motion_detected = self.motion_detector.detect(frame1, frame2)

            if motion_detected:
                self.no_motion_start_time = None
            else:
                if self.no_motion_start_time is None:
                    self.no_motion_start_time = time.time()
                elif (
                    time.time() - self.no_motion_start_time
                    >= Config.NO_MOTION_SEC_THRESHOLD
                ):
                    self.handle_no_motion(frame2)

            frame1 = frame2

            if ((time.time() - self.start_time) >= Config.MINUTES_BEFORE_STOPPING * 60):
                msg = f"{Config.MINUTES_BEFORE_STOPPING} minutes have passed. Stopping the script."
                print(msg)
                break

            if ((time.time() - last_heartbeat) >= Config.HEARTBEAT_SEC):
                msg = f'heartbeat - {time.time()}'
                print(msg)
                last_heartbeat = time.time()

            time.sleep(1 / Config.SAMPLING_RATE_FPS)


if __name__ == "__main__":
    script = MotionDetectionScript()
    script.run()
    msg = f"Reached timeout {Config.MINUTES_BEFORE_STOPPING} minutes"
    print(msg)
    Notifier.notify(msg)
