# Air Drawing Application

This project implements a real-time air drawing system using computer vision and hand gesture recognition. Users can draw, erase, pause, and refine geometric shapes in mid-air using only a webcam, without requiring any physical input device.

The system combines MediaPipe hand tracking with classical computer vision techniques such as trajectory smoothing, tracking-loss compensation, contour analysis, Hough Transform, and shape refinement. It supports the detection and correction of circles, lines, triangles, rectangles, and squares, providing a more accurate and visually appealing drawing experience.

The project was developed as part of the CS423/523 Computer Vision course.

## File structure

```text
air_drawing_clean_project/
├── config.py              # Constants and configuration values
├── gestures.py            # Hand gesture detection functions
├── drawing_utils.py       # Drawing helpers, smoothing, undo, screenshots
├── shape_refinement.py    # Shape detection and refinement logic
├── main.py                # Main webcam loop
└── README.md
```

## Run

Install the required libraries first:

```bash
pip install opencv-python mediapipe numpy
```

Then run:

```bash
python main.py
```

## Controls

- `q`: quit
- `c`: clear canvas
- `s`: refine current shape
- `d`: save screenshots
- `z`: undo
- `a`: show speed statistics
- `b/g/r/y/w`: change brush color
- `+` or `=`: increase brush/eraser thickness
- `-`: decrease brush/eraser thickness
