import cv2

GRAY_WIDTH = 320
PIXEL_DIFF = 25
MIN_CHANGED_RATIO = 0.002
MAX_SKIPPED_FRAMES = 15


class MotionGate:
    def __init__(self):
        self.reference = None
        self.skipped = 0

    def should_detect(self, frame):
        small = cv2.resize(frame, (GRAY_WIDTH, round(frame.shape[0] * GRAY_WIDTH / frame.shape[1])))
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        detect = (self.reference is None or self.skipped >= MAX_SKIPPED_FRAMES
                  or (cv2.absdiff(gray, self.reference) > PIXEL_DIFF).mean() > MIN_CHANGED_RATIO)
        if detect:
            self.reference, self.skipped = gray, 0
        else:
            self.skipped += 1
        return detect
