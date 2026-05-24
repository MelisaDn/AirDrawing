import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
from datetime import datetime

from config import (
    SAVE_DIR,
    MAX_PREDICT_FRAMES,
    LOST_FRAME_LIMIT,
    DRAW_START_DELAY,
)
from gestures import isOnlyIndexFingerUp, isPeaceSign, isOpenHand
from drawing_utils import smoothPoint, saveCanvasState
from shape_refinement import refineShape


# Stores recent fingertip positions for moving-average smoothing.
points_buffer = deque(maxlen=5)
trajectory_points = []
is_drawing_shape = False

# Open the default webcam.
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# MediaPipe Hands model configuration.
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# Canvas/state variables used during drawing, erasing, undo, and refinement.
canvas = None
canvas_history = []
prev_x, prev_y = None, None
draw_counter = 0
brush_color = (255, 0, 0)  # blue
brush_thickness = 5
eraser_thickness = 20
lost_frames = 0
speed_history = []
last_velocity = (0, 0)
show_speed_stats = False
avg_speed_text = ""
min_speed_text = ""
max_speed_text = ""

# Main webcam loop. All keyboard controls stay here because they modify live state variables.
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # If a hand is detected, process its landmarks and decide the active gesture.
    if result.multi_hand_landmarks:
        lost_frames = 0

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            x, y = smoothPoint(x, y, points_buffer)

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Open hand gesture activates erasing mode.
            if isOpenHand(hand_landmarks):
                draw_counter += 1

                cv2.putText(frame, "ERASING", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if draw_counter >= DRAW_START_DELAY:
                    if draw_counter == DRAW_START_DELAY:
                        saveCanvasState(canvas, canvas_history)
                        trajectory_points.clear()
                        trajectory_points.append((x, y))
                        prev_x, prev_y = x, y
                        continue

                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y),
                                 (0, 0, 0), eraser_thickness)

                    prev_x, prev_y = x, y

            # Only index finger up activates drawing mode.
            elif isOnlyIndexFingerUp(hand_landmarks):
                draw_counter += 1

                if draw_counter >= DRAW_START_DELAY:
                    cv2.putText(frame, "DRAWING", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if draw_counter == DRAW_START_DELAY:
                        saveCanvasState(canvas, canvas_history)
                        trajectory_points.clear()
                        trajectory_points.append((x, y))
                        prev_x, prev_y = x, y
                        continue

                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y),
                                 brush_color, brush_thickness)

                        last_velocity = (x - prev_x, y - prev_y)

                         # Hand speed (pixels/frame)
                        speed = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                        speed_history.append(speed)

                        # Display speed
                        # cv2.putText(frame,
                        #             f"Speed: {int(speed)}",
                        #             (20, 130),
                        #             cv2.FONT_HERSHEY_SIMPLEX,
                        #             1,
                        #             (255, 255, 255),
                        #             2)

                    trajectory_points.append((x, y))
                    prev_x, prev_y = x, y

                else:
                    cv2.putText(frame, "READY...", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    prev_x, prev_y = None, None

            # Peace sign pauses drawing/erasing.
            elif isPeaceSign(hand_landmarks):
                draw_counter = 0
                prev_x, prev_y = None, None
                points_buffer.clear()

                cv2.putText(frame, "PAUSED", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

            else:
                draw_counter = 0
                prev_x, prev_y = None, None
                points_buffer.clear()

                cv2.putText(frame, "PAUSED", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    else:
        # If the hand disappears, continue briefly using the last velocity.
        lost_frames += 1

        # Predict short missing path when MediaPipe temporarily loses the hand
        if (
            lost_frames <= MAX_PREDICT_FRAMES
            and prev_x is not None
            and prev_y is not None
            and draw_counter >= DRAW_START_DELAY
        ):
            vx, vy = last_velocity

            pred_x = int(prev_x + vx)
            pred_y = int(prev_y + vy)

            pred_x = max(0, min(w - 1, pred_x))
            pred_y = max(0, min(h - 1, pred_y))

            cv2.line(canvas, (prev_x, prev_y), (pred_x, pred_y),
                     brush_color, brush_thickness)

            trajectory_points.append((pred_x, pred_y))
            prev_x, prev_y = pred_x, pred_y

        elif lost_frames > LOST_FRAME_LIMIT:
            draw_counter = 0
            prev_x, prev_y = None, None
            points_buffer.clear()

    output = cv2.addWeighted(frame, 0.7, canvas, 0.7, 0)

    if show_speed_stats:

        cv2.putText(output,
                    avg_speed_text,
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        cv2.putText(output,
                    min_speed_text,
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        cv2.putText(output,
                    max_speed_text,
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)
    cv2.imshow("Air Drawing", output)
    cv2.imshow("Canvas Only", canvas)

    # Keyboard shortcuts: quit, clear, save, undo, refine, colors, and thickness.
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        canvas = np.zeros_like(frame)
        prev_x, prev_y = None, None
        points_buffer.clear()
        trajectory_points.clear()
        draw_counter = 0

    if key == ord("a"):

        if len(speed_history) > 0:

            avg_speed = np.mean(speed_history)
            min_speed = np.min(speed_history)
            max_speed = np.max(speed_history)

            avg_speed_text = f"Average Speed: {avg_speed:.2f}"
            min_speed_text = f"Min Speed: {min_speed:.2f}"
            max_speed_text = f"Max Speed: {max_speed:.2f}"

            show_speed_stats = True

            speed_history.clear()


    if key == ord("b"):
        brush_color = (255, 0, 0)   # blue

    if key == ord("g"):
        brush_color = (0, 255, 0)   # green

    if key == ord("r"):
        brush_color = (0, 0, 255)   # red

    if key == ord("y"):
        brush_color = (0, 255, 255) # yellow

    if key == ord("w"):
        brush_color = (255, 255, 255) # white

    if key == ord("s"):
        saveCanvasState(canvas, canvas_history)
        canvas = refineShape(canvas, trajectory_points, brush_color, brush_thickness)
        trajectory_points.clear()
        prev_x, prev_y = None, None
        points_buffer.clear()
        draw_counter = 0

    if key == ord("d"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = os.path.join(SAVE_DIR, f"air_drawing_output_{timestamp}.png")
        canvas_path = os.path.join(SAVE_DIR, f"canvas_only_{timestamp}.png")

        cv2.imwrite(output_path, output)
        cv2.imwrite(canvas_path, canvas)

        print("Saved screenshots:")
        print(output_path)
        print(canvas_path)

    if key == ord("z"):
        if len(canvas_history) > 0:
            canvas = canvas_history.pop()
            trajectory_points.clear()
            prev_x, prev_y = None, None
            points_buffer.clear()
            draw_counter = 0
            print("Undo")
        else:
            print("Nothing to undo")

    if key == ord("+") or key == ord("="):
        if isOpenHand(hand_landmarks):
            eraser_thickness += 2
        else:
            brush_thickness += 1

    if key == ord("-"):
        if isOpenHand(hand_landmarks):
            eraser_thickness = max(5, eraser_thickness - 2)
        else:
            brush_thickness = max(1, brush_thickness - 1)

cap.release()
cv2.destroyAllWindows()
