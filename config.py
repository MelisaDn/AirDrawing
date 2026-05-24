import os

# Folder used when the user presses the screenshot/save key.
SAVE_DIR = "air_drawing_screenshots"
os.makedirs(SAVE_DIR, exist_ok=True)

# Number of frames to predict when MediaPipe temporarily loses the hand.
MAX_PREDICT_FRAMES = 5

# After this many lost frames, drawing state is reset.
LOST_FRAME_LIMIT = 8

# Drawing/erasing starts only after the gesture is stable for this many frames.
DRAW_START_DELAY = 5
