"""
This script defines a bot that plays Chrome's Dino Game.
"""

import dataclasses
import platform
import time
import webbrowser
from importlib import resources
from typing import Optional, Tuple

import cv2
import mss
import numpy as np
import pyautogui


@dataclasses.dataclass
class DinoParams:
    """
    Dataclass holding configuration parameters for the Dino game bot.

    Attributes:
        confidence_threshold (float): Confidence threshold for template matching.
        day_threshold (float): A ratio threshold to distinguish daytime from nighttime (based on 'white sky' pixels).
        obstacle_min_contrast (float): Minimum contrast ratio threshold for detecting an obstacle.
        obstacle_max_contrast (float): Maximum contrast ratio threshold for detecting an obstacle.
        late_game_time (float): Number of seconds after which the bot considers it 'late game'.
        idle_reset_time (float): Number of seconds before the bot presses space to reset if no obstacles are detected.
        duck_time (float): Duration (in seconds) to hold the 'down' key for flying obstacles.
        post_jump_duck_sleep (float): Short pause after jumping, before ducking, in late game.
        init_scale_w (float): Horizontal scaling factor (width multiplier) used to detect obstacles in the early game.
                              Also serves as the minimal scale in our linear interpolation.
        late_scale_w (float): Horizontal scaling factor (width multiplier) used to detect obstacles in the late game.
                              Also serves as the maximum scale in our linear interpolation.
    """

    confidence_threshold: float = 0.8
    day_threshold: float = 0.5
    obstacle_min_contrast: float = 0.1
    obstacle_max_contrast: float = 0.9
    late_game_time: float = 30.0
    idle_reset_time: float = 7.0
    duck_time: float = 0.4
    post_jump_duck_sleep: float = 0.02
    init_scale_w: float = 1.5
    late_scale_w: float = 4.0


class DinoBot:
    """
    A class-based bot that automates the Chrome Dino game.

    The bot continuously captures the game screen, detects whether it is day or night,
    locates obstacles, and issues appropriate keyboard actions (jump or duck).
    """

    def __init__(self, screen_id: int, params: DinoParams, debug: bool = False):
        """
        Initialize the DinoBot by loading templates, resizing them to match
        the current screen resolution, and setting up initial state.

        Args:
            screen_id (int): Index of the monitor to capture (as recognized by mss).
            params (DinoParams): Configuration parameters for the Dino bot.
            debug (bool): Enable debug mode.
        """
        self.screen_id = screen_id
        self.params = params
        self.debug = debug

        # Original resolution for which templates were designed
        self.orig_width = 1920
        self.orig_height = 1080

        # Load day/night templates
        with resources.path("dino_chrome_bot.templates", "dino_day.png") as img_path:
            dino_day_template_filepath = img_path

        with resources.path("dino_chrome_bot.templates", "dino_night.png") as img_path:
            dino_night_template_filepath = img_path

        original_day = cv2.imread(dino_day_template_filepath, cv2.IMREAD_COLOR)
        original_night = cv2.imread(dino_night_template_filepath, cv2.IMREAD_COLOR)

        # Resize templates to match current screen's resolution
        self.template_day = self._resize_template(original_day)
        self.template_night = self._resize_template(original_night)

        # Internal state for the Dino’s position and template size
        self.skip_initialize = False
        self.top_left = (0, 0)
        self.template_h = 0
        self.template_w = 0

        self.last_button_push = time.perf_counter()
        self.start_time = time.perf_counter()

    def _resize_template(self, template: np.ndarray) -> Optional[np.ndarray]:
        """
        Resize the given template to match the current screen resolution
        based on the original 1920x1080 sizing.

        Args:
            template (np.ndarray): The loaded BGR template image.

        Returns:
            np.ndarray or None: Resized template, or None if input is None.
        """
        if template is None:
            return None

        # Get current monitor resolution to compute scale
        with mss.mss() as sct:
            mon = sct.monitors[self.screen_id]
            screen_width = mon["width"]
            screen_height = mon["height"]

        scale_w = screen_width / self.orig_width
        scale_h = screen_height / self.orig_height

        new_width = int(template.shape[1] * scale_w)
        new_height = int(template.shape[0] * scale_h)

        # Ensure new dimensions are not zero
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        resized = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized

    def open_chrome(self, url: str, app_path: Optional[str] = None) -> None:
        """
        Open Google Chrome with the given URL using a specified browser path,
        or a best-guess path depending on the platform (Windows, Mac, Linux).

        Args:
            url (str): The URL to open (e.g. 'chrome://dino/').
            app_path (str, optional): Custom path or command to Chrome.
                                      If None, attempts an OS-based guess.
        """
        if app_path is None:
            system_name = platform.system().lower()
            if "win" in system_name:
                app_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe %s"
            elif "darwin" in system_name:
                app_path = "open -a /Applications/Google\\ Chrome.app %s"
            elif "linux" in system_name:
                app_path = "google-chrome %s"
            else:
                raise ValueError("Unrecognized OS. Provide a valid app_path for Chrome.")

        try:
            webbrowser.get(app_path).open(url)
        except webbrowser.Error as e:
            raise ValueError(
                f"Could not open {url} using '{app_path}'. "
                "Please ensure Chrome is installed or provide a valid 'app_path'."
            ) from e

    def verify_chrome_dino(self, wait_seconds: float = 5.0) -> bool:
        """
        Wait a few seconds, then attempt to find the Dino in the screenshot.
        If not found, warns user to open 'chrome://dino/' manually.

        Args:
            wait_seconds (float): How many seconds to wait before checking.

        Returns:
            bool: True if Dino was found, False otherwise.
        """
        time.sleep(wait_seconds)

        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[self.screen_id])
            frame = np.array(screenshot)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            found, max_val, *_ = self._find_dinosaur(frame_bgr)

        if not found:
            print("Warning: Could not detect the Dino game in Chrome.")
            print(
                "Please open 'chrome://dino/' manually if you haven't already and place Chrome in the correct screen."
            )
            print("Make sure to click on Chrome's Dino tab so that bot's key inputs are registered.")
            return False
        return True

    def _find_dinosaur(self, frame_bgr: np.ndarray) -> Tuple[bool, float, Tuple[int, int], int, int]:
        """
        Attempt to find the dinosaur in the frame using both day and night templates.

        Args:
            frame_bgr (np.ndarray): Current BGR frame from the screen.

        Returns:
            A tuple (found, max_val, top_left, template_h, template_w), where:
             - found (bool): Whether the dino was found above the confidence threshold.
             - max_val (float): Best template matching score found.
             - top_left (Tuple[int, int]): Coordinates of top-left corner of the best match.
             - template_h (int): The template's height (pixels).
             - template_w (int): The template's width (pixels).
        """
        found = False
        max_val = 0.0
        top_left = (0, 0)
        template_h = 0
        template_w = 0

        for template in (self.template_night, self.template_day):
            if template is None:
                continue

            result = cv2.matchTemplate(frame_bgr, template, cv2.TM_CCOEFF_NORMED)
            _, local_max_val, _, local_max_loc = cv2.minMaxLoc(result)

            if local_max_val > max_val:
                max_val = local_max_val
                top_left = local_max_loc
                template_h, template_w, _ = template.shape

        if max_val >= self.params.confidence_threshold:
            found = True
        return found, max_val, top_left, template_h, template_w

    def _is_day_scene(self, frame_gray: np.ndarray) -> bool:
        """
        Determine if it's 'daytime' or 'nighttime' by sampling the sky area above the dinosaur.

        Args:
            frame_gray (np.ndarray): Grayscale version of the current screenshot.

        Returns:
            bool: True if the scene is considered daytime; False otherwise.
        """
        x, y = self.top_left
        sky_y_start = max(int(y - self.template_h), 0)
        sky_y_end = int(y)

        sky_region = frame_gray[sky_y_start:sky_y_end, :]
        if sky_region.size == 0:
            return False

        white_ratio = (sky_region == 255).sum() / sky_region.size
        return white_ratio > self.params.day_threshold

    def _threshold_obstacle_frames(
        self, frame_gray: np.ndarray, is_day: bool, scale_w: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and threshold the middle and bottom obstacle regions using the
        specified horizontal scaling factor.

        Args:
            frame_gray (np.ndarray): Grayscale screenshot image.
            is_day (bool): Indicates if the game scene is day or night.
            scale_w (float): Current horizontal width multiplier for detecting obstacles.

        Returns:
            (np.ndarray, np.ndarray): Thresholded frames for the middle and bottom regions.
        """
        x, y = self.top_left

        new_top_left = y - 0.5 * self.template_h
        new_h = 1.5 * self.template_h

        # Calculate bounding box slices
        middle_obstacle_frame = frame_gray[
            int(new_top_left + new_h / 3) : int(new_top_left + 2 * new_h / 3),
            int(x + (scale_w - 1) * self.template_w) : int(x + scale_w * self.template_w),
        ]
        bottom_obstacle_frame = frame_gray[
            int(new_top_left + 2 * new_h / 3) : int(new_top_left + new_h),
            int(x + (scale_w - 1) * self.template_w) : int(x + scale_w * self.template_w),
        ]

        if is_day:
            # Daytime threshold
            _, middle_thresh = cv2.threshold(middle_obstacle_frame, 180, 255, cv2.THRESH_BINARY)
            _, bottom_thresh = cv2.threshold(bottom_obstacle_frame, 180, 255, cv2.THRESH_BINARY)
        else:
            # Nighttime with Otsu
            _, middle_thresh = cv2.threshold(middle_obstacle_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, bottom_thresh = cv2.threshold(bottom_obstacle_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return middle_thresh, bottom_thresh

    def _compute_contrasts(self, *frames: np.ndarray) -> Tuple[float, ...]:
        """
        Compute the ratio of white pixels to total pixels for multiple thresholded frames.

        Args:
            *frames (np.ndarray): One or more thresholded frames.

        Returns:
            Tuple[float, ...]: A tuple of contrast ratios, one per frame.
        """
        contrasts = []
        for f in frames:
            if f.size == 0:
                contrasts.append(0.0)
            else:
                ratio = (f == 255).sum() / f.size
                contrasts.append(ratio)
        return tuple(contrasts)

    def _show_debug_frames(self, *frames: Tuple[np.ndarray, ...]) -> None:
        """
        Display frames in separate windows for debugging and overlay their contrast.

        Args:
            *frames (np.ndarray): Frames to show (e.g. middle obstacle, bottom obstacle).
        """
        window_names = ["middle_obstacle_frame", "bottom_obstacle_frame"]
        for name, frame in zip(window_names, frames):
            cv2.imshow(name, frame)
        cv2.waitKey(1)

    def _get_dynamic_scale_w(self, elapsed_time: float) -> float:
        """
        Computes a linearly increasing scale_w from init_scale_w to late_scale_w
        over the duration of late_game_time.

        Args:
            elapsed_time (float): How many seconds have passed since game started.

        Returns:
            float: The appropriate scale_w for the current elapsed time.
        """
        if elapsed_time >= self.params.late_game_time:
            # Once we've reached or passed late_game_time, return the max scale
            return self.params.late_scale_w
        # Otherwise, linearly interpolate
        ratio = elapsed_time / self.params.late_game_time
        return self.params.init_scale_w + ratio * (self.params.late_scale_w - self.params.init_scale_w)

    def play(self) -> None:
        """
        Main game loop that continuously captures the screen, detects obstacles, and
        simulates keystrokes (jump or duck) to avoid them.
        """
        with mss.mss() as sct:
            while True:
                screenshot = sct.grab(sct.monitors[self.screen_id])
                frame = np.array(screenshot)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                cv2.imshow(
                    "Screen being captured",
                    cv2.resize(frame_bgr, (frame_bgr.shape[1] // 6, frame_bgr.shape[0] // 6)),
                )
                cv2.waitKey(1)

                # 1. Use template matching to find initial dinosaur location
                if not self.skip_initialize:
                    found, max_val, self.top_left, self.template_h, self.template_w = self._find_dinosaur(frame_bgr)
                    if found:
                        pyautogui.press("space")
                        self.skip_initialize = True
                        self.last_button_push = time.perf_counter()
                        self.start_time = time.perf_counter()
                        print("Dino found. Let's play!")
                    continue

                # 2. Obstacle detection
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                day_scene = self._is_day_scene(frame_gray)

                elapsed_time = time.perf_counter() - self.start_time
                # Dynamically compute scale_w (linear interpolation)
                scale_w = self._get_dynamic_scale_w(elapsed_time)

                middle_thresh_frame, bottom_thresh_frame = self._threshold_obstacle_frames(
                    frame_gray, day_scene, scale_w
                )
                middle_contrast, bottom_contrast = self._compute_contrasts(middle_thresh_frame, bottom_thresh_frame)

                if self.debug:
                    self._show_debug_frames(middle_thresh_frame, bottom_thresh_frame)

                now = time.perf_counter()
                is_obstacle_middle = (
                    self.params.obstacle_min_contrast < middle_contrast < self.params.obstacle_max_contrast
                )
                is_obstacle_bottom = (
                    self.params.obstacle_min_contrast < bottom_contrast < self.params.obstacle_max_contrast
                )

                # 3. Input action
                if is_obstacle_bottom:
                    # Late game => jump, then duck
                    if elapsed_time >= self.params.late_game_time:
                        pyautogui.keyDown("space")
                        pyautogui.keyUp("space")
                        time.sleep(self.params.post_jump_duck_sleep)
                        pyautogui.keyDown("down")
                        pyautogui.keyUp("down")
                    else:
                        # Early game => just jump
                        pyautogui.press("space")
                    self.last_button_push = now

                elif is_obstacle_middle:
                    # If obstacle is in the middle region (bird) => duck briefly
                    pyautogui.keyDown("down")
                    time.sleep(self.params.duck_time)
                    pyautogui.keyUp("down")
                    self.last_button_push = now

                # If idle too long, press space (restart game)
                if (now - self.last_button_push) > self.params.idle_reset_time:
                    print("Game frozen. Pressing 'space' ...")
                    pyautogui.press("space")
                    self.start_time = time.perf_counter()
                    self.last_button_push = now


def main():
    """
    Script entry point.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Class-based Chrome Dino bot.")
    parser.add_argument(
        "--screen_id",
        type=int,
        default=1,
        help="Monitor index as recognized by mss (1-based).",
    )
    parser.add_argument(
        "--open_chrome",
        type=bool,
        default=True,
        help="True/False to open Chrome automatically.",
    )
    parser.add_argument("--chrome_path", type=str, default=None, help="Path or command to Chrome.")
    parser.add_argument("--debug", type=bool, default=False, help="True/False to enable debug mode.")

    args = parser.parse_args()
    print(args)

    # Define and override DinoParams here as needed
    dino_params = DinoParams()

    # Create bot with chosen screen & parameters
    bot = DinoBot(screen_id=args.screen_id, params=dino_params, debug=args.debug)

    # Optionally open Chrome
    if args.open_chrome:
        bot.open_chrome("chrome://dino/", app_path=args.chrome_path)
        # Check if Dino is actually opened
        bot.verify_chrome_dino(wait_seconds=5.0)

    # Start game loop
    bot.play()


if __name__ == "__main__":
    main()
