import os
import time
from datetime import datetime
import pytz
import cv2
import numpy as np
import pyautogui
from pync import Notifier
from collections import deque, OrderedDict, defaultdict
from PIL import UnidentifiedImageError
from threading import Thread, Event



class Config:
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M")
    SCREENSHOT_SAMPLING_RATE_FPS = 200.0
    SAMPLING_RATE_FPS = 100.0
    TIME_LIMIT = 30000
    NO_MOTION_THRESHOLD = 0.4
    NO_MOTION_CLICK_THRESHOLD = 0.1
    TEMPLATE_PATH = '/Users/ericxu/Documents/Jupyter/mahjong/templates'
    SCREENSHOT_PATH = '/Users/ericxu/Documents/Jupyter/mahjong/auto_screenshots'
    
    TEMPLATE_BOUNDARY_MAP = {
        'GameScreen': {"x_min": 1140, "y_min": 1260},
        'NextGame': {"x_min": 1720, "y_min": 1220},
        'Pong': {"x_min": 1930, "y_min": 820},
        'Kong': {"x_min": 1930, "y_min": 820},
        'Chow': {"x_min": 1930, "y_min": 820},
        'YourDiscard': {"x_min": 2000, "y_min": 820},
        'KongPong': {"x_min": 1750, "y_min": 820},
        'PongChow': {"x_min": 1750, "y_min": 820},
        'NotFullScreen': {"x_max": 250, "y_max": 100}
        "DetermineWinner": {"x_min": 0, "y_min": 400, "x_max": 1250 ,"y_max":800},
        "DetermineSelfPick": {"x_min": 1650, "y_min": 400 ,"y_max":1200},
        "DetermineWinnerFan": {"x_min": 1650, "y_min": 400 ,"y_max":1200},
    }

    # Based on mac 13" screen resolution: 1280x800
    CLICK_COORDINATES = {
        "cancel": (1220, 430), "wall_tile": (1100, 540),
        "D13": (980, 540), "D12": (915, 540), "D11": (850, 540),
        "D10": (785, 540), "D9": (720, 540), "D8": (655, 540),
        "D7": (590, 540), "D6": (525, 540), "D5": (460, 540),
        "D4": (395, 540), "D3": (330, 540), "D2": (265, 540),
        "D1": (200, 540), "accept": (1100, 430), "accept_left": (980, 430),
        "x2": (1241, 151), "x": (1118, 200), "close_ad": (1198, 147),
        "exit_ad": (1205, 191), "exit_ad2": (1202, 197), "exit_ad3": (1243, 141),
        "exit_ad4": (668, 604), "next_game": (893, 626), "draw_game": (618, 450),
    }

    ACTION_SCREEN_CLICK_ORDER = {
        "NextGame": ['next_game'],
        "Draw": ['draw_game'],
        "YourDiscard": ["wall_tile","D13","D12","D11","D10","D9","D8","D7","D6","D5","D4","D3","D2", "D1"],
        "Chow": ['accept','reject'],
        "ChowSelection": ["wall_tile","D13","D12","D11","D10","D9","D8","D7","D6","D5","D4","D3","D2", "D1"],
        "Pong": ['accept','reject'],
        "Kong": ['accept','reject'],
        "PongChow": ['accept_left', 'accept','cancel'],
        "KongPong": ['accept_left', 'accept','cancel'],
        "Ad": ["exit_ad","x2","x","close_ad","exit_ad4","exit_ad3","exit_ad2"],
    }

    GAME_ACTION_SCREENS = ['YourDiscard', 'Chow','Pong','Kong','ChowSelection','KongPong', 'PongChow', 'Draw']

    NEXT_GAME_CHILDREN = ["DetermineWinner","DetermineSelfPick","DetermineWinnerFan"]



class Utils:
    @staticmethod
    def is_screen(frame, type='GameScreen', threshold=0.97, xy_search_boundaries=None):
        if xy_search_boundaries is None:
            xy_search_boundaries = {}


        xy_search_boundaries.update(Config.TEMPLATE_BOUNDARY_MAP.get(type, {}))
        x_min = xy_search_boundaries.get("x_min", 0)
        x_max = xy_search_boundaries.get("x_max", frame.shape[1])
        y_min = xy_search_boundaries.get("y_min", 0)
        y_max = xy_search_boundaries.get("y_max", frame.shape[0])

        f = frame[y_min:y_max, x_min:x_max]
        fg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        folder_path = os.path.join(Config.TEMPLATE_PATH, type)
        png_files = (f for f in os.listdir(folder_path) if f.endswith('.png'))

        for template_path in png_files:
            tg = cv2.imread(os.path.join(
                folder_path, template_path), cv2.IMREAD_GRAYSCALE)
            matched_result = cv2.matchTemplate(fg, tg, cv2.TM_CCOEFF_NORMED)
            match_found = np.any(matched_result >= threshold)

            if match_found:
                yloc, xloc = np.where(matched_result >= threshold)
                print(f'Match found with {type}/{template_path}')
                return True, [int(min(xloc)), int(max(xloc)), int(min(yloc)), int(max(yloc))]

        return False, [0, 0, 0, 0]

    @staticmethod
    def save_screenshot(frame, prefix=''):
        ms_ts_id = int(time.time() * 1000)  # millisecond timestamp id
        dir_path = os.path.join(Config.SCREENSHOT_PATH, Config.TIMESTAMP)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f'{prefix}_{ms_ts_id}.png')
        cv2.imwrite(path, frame)
        print(f"{prefix} Screenshot saved @ {ms_ts_id}.png")

    @staticmethod
    def current_time_to_pst():
        utc_dt = datetime.utcfromtimestamp(time.time())
        pst_tz = pytz.timezone('America/Los_Angeles')
        pst_dt = pytz.utc.localize(utc_dt).astimezone(pst_tz)
        return str(pst_dt)[0:19]


class ScreenshotCapturer:
    @staticmethod
    def capture():
        try:
            screenshot = pyautogui.screenshot()
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Screenshot capture failed: {e}")
            time.sleep(5)
            return None


class GameFrameQueue:
    def __init__(self, maxlen=2):
        self.queue = deque(maxlen=maxlen)

    def enqueue(self, frame):
        if isinstance(frame, np.ndarray):
            self.queue.append(frame)
        else:
            raise ValueError("Only numpy arrays are allowed.")

    def length(self):
        return len(self.queue)

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0:
                index += len(self.queue)
            if index < 0 or index >= len(self.queue):
                raise IndexError("Index out of range")
            return self.queue[index]
        else:
            raise TypeError("Index must be an integer")

    def save_screenshot(self, frame):
        ms_ts_id = int(time.time() * 1000)
        path = os.path.join(
            f'{Config.SCREENSHOT_PATH}/{Config.TIMESTAMP}/DAEMON_{ms_ts_id}.png')
        cv2.imwrite(path, frame)
        msg = f"Screenshot saved @ DAEMON_{ms_ts_id}.png"
        print(msg)
        Notifier.notify(msg)

class MotionDetector:
    def __init__(self, threshold=10):
        self.threshold = threshold

    def detect(self, frame1, frame2, screen_type='WallCount'):
        f1, f2 = frame1.copy(), frame2.copy()       # masked frames for motion capture
        f1_, f2_ = frame1.copy(), frame2.copy()     # masked frames for screenshots

        if screen_type == 'WallCount':
            self.threshold = 1  # make highly sensitive to wall count changes
            f1[:1190, :] = f2[:1190, :] = 0
            f1[:, 1450:] = f2[:, 1450:] = 0
            f1[1190:1400, :1150] = f2[1190:1400, :1150] = 0

            f1_[1390:, :, :] = f2_[1390:, :, :] = 0
            f1_[0:210, :, :] = f2_[0:210, :, :] = 0
            f1_[1190:1400, 0:1170, :] = f2_[1190:1400, 0:1170, :] = 0
        elif screen_type == 'GameScreenAction':
            self.threshold=20   # less sensitive to ignore mouse pointer
            f1[0:210, :, :] = f2[0:210, :, :] = 0
            f1[950:, :, :] = f2[950:, :, :] = 0

            f1_[1390:, :, :] = f2_[1390:, :, :] = 0
            f1_[0:210, :, :] = f2_[0:210, :, :] = 0
            f1_[1190:1400, 0:1170, :] = f2_[1190:1400, 0:1170, :] = 0
        elif screen_type == 'other':
            self.threshold = 30
            f1[1200:], f1[:210] = 0, 0
            f2[1200:], f2[:210] = 0, 0
 

        diff = cv2.absdiff(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY))
        _, diff_thresh = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours) > 0, f1_, f2_


class ClickMotionDetector(MotionDetector):
    def detect_after_click(self, location, type='GameScreen'):
        frame1 = ScreenshotCapturer.capture()
        pyautogui.moveTo(location)
        # print(f"Moved mouse to {location}")
        pyautogui.click()
        # time.sleep(Config.NO_MOTION_CLICK_THRESHOLD)
        frame2 = ScreenshotCapturer.capture()
        return self.detect(frame1, frame2, type=type)

class GameScreenshotSaveTask(Thread):
    def __init__(self, game_frame_queue, motion_detector, stop_event):
        super().__init__()
        self.game_frame_queue = game_frame_queue
        self.motion_detector = motion_detector
        self.stop_event = stop_event
        self.last_motion_time = time.time()
        self.motion_detected = False

    def run(self):
        while not self.stop_event.is_set():
            frame = ScreenshotCapturer.capture()
            is_game_frame, match_locations = Utils.is_screen(
                frame, type='GameScreen')
            if is_game_frame:
                game_frame_queue.enqueue(frame)
                if self.game_frame_queue.length() > 1:
                    detected, f1, f2 = self.motion_detector.detect(self.game_frame_queue[0], self.game_frame_queue[1])
                    if detected:
                        self.game_frame_queue.save_screenshot(f2)
                elif self.game_frame_queue.length() == 1:
                    self.game_frame_queue.save_screenshot(frame)
            time.sleep(1 / Config.SCREENSHOT_SAMPLING_RATE_FPS)


# Main script
if __name__ == "__main__":
    msg = f'Starting script @ {Utils.current_time_to_pst()}'
    print(msg)
    # Notifier.notify(msg)
    start_time = time.time()

    game_frame_queue = GameFrameQueue()
    motion_detector = MotionDetector()
    stop_event = Event()

    # Start the background motion detection task
    game_screenshot_save_task = GameScreenshotSaveTask(
        game_frame_queue, motion_detector, stop_event)
    game_screenshot_save_task.daemon = True
    game_screenshot_save_task.start()

    non_game_frame_queue = GameFrameQueue()
    click_motion_detector = ClickMotionDetector()
    is_your_discard_or_chow_selection = 0
    motion_detected = False
    last_motion_time = time.time()
    game_action_frame_queue = GameFrameQueue()
    
    # Main loop
    while (time.time() - start_time) < Config.TIME_LIMIT:
        frame = ScreenshotCapturer.capture()
        if frame is None:
            continue

        your_discard, _your_discard_loc = Utils.is_screen(frame, type='YourDiscard')
        not_full_screen, _not_full_screen_loc = Utils.is_screen(frame, type='NotFullScreen')
        game_screen, _game_screen_loc = Utils.is_screen(frame, type='GameScreen')
        next_game, _next_game_loc = Utils.is_screen(frame, type='NextGame')
        ad, _ad_loc = Utils.is_screen(frame, type='Ad')
        draw, _draw_loc = Utils.is_screen(frame, type='Draw')

        # game screen
        if not_full_screen:
            print(
                f'is_not_full_screen_frame_locations = {is_not_full_screen_frame_locations}')
            Notifier.notify('full screen not detected')
            time.sleep(5)
        elif game_screen:
            game_action_frame_queue.enqueue(frame)
            for _type in GAME_ACTION_SCREENS:
                r, l = Utils.is_screen(frame, type=_type)
                if r:
                    msg = f'game action screen {r} found @ {l}'
                    Notifier.notify(msg)
                    print(msg)

                    for t, c in Config.YOUR_DISCARD_CHOW_SELECTION_CLICK_COORDINATES.items():
                        if click_motion_detector.detect_after_click(c, type='DiscardChowSelection'):
                            msg = f"Motion detected after clicking {t} at {c}"
                            # Notifier.notify(msg)
                            break


        if game_screen:
            game_frame_queue.enqueue(frame)
            print(
                f'is_game_frame / match_locations = {is_game_frame} / {match_locations} \n')

            is_game_frame, match_locations = Utils.is_screen(frame, type='YourDiscard')

            # Check if no motion
            if not motion_detected and (time.time() - last_motion_time) > Config.NO_MOTION_THRESHOLD:
                msg = f"No motion detected for {Config.NO_MOTION_THRESHOLD} seconds"
                print(msg)

                your_turn_map = defaultdict(str)

                for tile in Config.YOUR_TURN_TILES:
                    found, loc = Utils.is_screen(frame, type=tile)
                    your_turn_map[tile] = [found, loc]
                    if found:
                        Utils.save_screenshot(frame, prefix=f'{tile}_')
                        if tile in ['YourDiscard', 'ChowSelection']:
                            is_your_discard_or_chow_selection = 1
                        break

                results = [v[0] for k, v in your_turn_map.items()]
                if max(results) == True:
                    # Notifier.notify('your turn detected')

                    print(results)
                    your_turn_map.clear()

                    if is_your_discard_or_chow_selection == 1:
                        is_your_discard_or_chow_selection = 0       # reset

                        for t, c in Config.YOUR_DISCARD_CHOW_SELECTION_CLICK_COORDINATES.items():
                            if click_motion_detector.detect_after_click(c, type='DiscardChowSelection'):
                                msg = f"Motion detected after clicking {t} at {c}"
                                # Notifier.notify(msg)
                                break
                    else:
                        for t, c in Config.YOUR_TURN_CLICK_COORDINATES.items():
                            if click_motion_detector.detect_after_click(c, type='other'):
                                msg = f"Motion detected after clicking {t} at {c}"
                                # Notifier.notify(msg)
                                break
            else:
                frame = ScreenshotCapturer.capture()
                is_draw_frame, match_locations = Utils.is_screen(
                    frame, type='Draw')
                t = 'draw_game'
                c = Config.CLICK_COORDINATES[t]
                if click_motion_detector.detect_after_click(c, type='other'):
                    msg = f"Motion detected after clicking {t} at {c}"
                    # Notifier.notify(msg)

        elif is_not_full_screen_frame:
            print(
                f'is_not_full_screen_frame_locations = {is_not_full_screen_frame_locations}')
            Notifier.notify('full screen not detected')
            time.sleep(5)
        else:
            non_game_frame_queue.enqueue(frame)
            is_next_game_frame, match_locations = Utils.is_screen(
                frame, type='NextGame')
            if is_next_game_frame:
                Utils.save_screenshot(frame, prefix='NextGame_')
                Notifier.notify('next game screen detected')
            else:
                is_ad_frame, match_locations = Utils.is_screen(
                    frame, type='Ad')
                if is_ad_frame:
                    Utils.save_screenshot(frame, prefix='Ad_')
                    Notifier.notify('ad screen detected')
                else:
                    Utils.save_screenshot(frame, prefix='Unknown_')
                    Notifier.notify('unknown screen detected')

            if non_game_frame_queue.length() > 1:
                if motion_detector.detect(non_game_frame_queue[0], non_game_frame_queue[1], type='other'):
                    last_motion_time = time.time()
                    motion_detected = True
                else:
                    motion_detected = False

            if not motion_detected and (time.time() - last_motion_time) > Config.NO_MOTION_THRESHOLD:
                print(
                    f"No motion detected for {Config.NO_MOTION_THRESHOLD} seconds.")
                Notifier.notify(
                    f"No motion detected for {Config.NO_MOTION_THRESHOLD} seconds.")
                for t, c in Config.OTHER_CLICK_COORDINATES.items():
                    if click_motion_detector.detect_after_click(c, type='other'):
                        Notifier.notify(
                            f"Motion detected after clicking {t} at {c}")
                        break

        time.sleep(1 / Config.SAMPLING_RATE_FPS)

    # Stop the background task and wait for it to finish
    stop_event.set()
    game_screenshot_save_task.join()

    msg = f'Ending script @ {Utils.current_time_to_pst()}'
    print(msg)
    Notifier.notify(msg)