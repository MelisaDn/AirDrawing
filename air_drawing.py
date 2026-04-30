import cv2
import mediapipe as mp
import numpy as np
from collections import deque

points_buffer = deque(maxlen=5)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

canvas = None
prev_x, prev_y = None, None
draw_counter = 0
brush_color = (255, 0, 0)  # blue
brush_thickness = 5
eraser_thickness = 20
lost_frames = 0

LOST_FRAME_LIMIT = 5
DRAW_START_DELAY = 5



def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


def isFingerExtended(lm, tip_id, pip_id, mcp_id):
    tip_to_wrist = distance(lm[tip_id], lm[0])
    pip_to_wrist = distance(lm[pip_id], lm[0])
    mcp_to_wrist = distance(lm[mcp_id], lm[0])

    return tip_to_wrist > pip_to_wrist and tip_to_wrist > mcp_to_wrist


def isOnlyIndexFingerUp(hand_landmarks):
    lm = hand_landmarks.landmark

    index_extended = isFingerExtended(lm, 8, 6, 5)
    middle_extended = isFingerExtended(lm, 12, 10, 9)
    ring_extended = isFingerExtended(lm, 16, 14, 13)
    pinky_extended = isFingerExtended(lm, 20, 18, 17)

    return index_extended and not middle_extended and not ring_extended and not pinky_extended


def isPeaceSign(hand_landmarks):
    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y

    return index_up and middle_up and ring_down and pinky_down


def isOpenHand(hand_landmarks):
    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    return index_up and middle_up and ring_up and pinky_up


def smoothPoint(x, y, points_buffer):
    points_buffer.append((x, y))

    avg_x = int(sum(p[0] for p in points_buffer) / len(points_buffer))
    avg_y = int(sum(p[1] for p in points_buffer) / len(points_buffer))

    return avg_x, avg_y


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        lost_frames = 0
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            x, y = smoothPoint(x, y, points_buffer)

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            if isOpenHand(hand_landmarks):
                draw_counter += 1

                cv2.putText(frame, "ERASING", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if draw_counter >= DRAW_START_DELAY:
                    if draw_counter == DRAW_START_DELAY:
                        prev_x, prev_y = x, y
                        continue

                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y),
                                 (0, 0, 0), eraser_thickness)

                    prev_x, prev_y = x, y

            elif isOnlyIndexFingerUp(hand_landmarks):
                draw_counter += 1

                if draw_counter >= DRAW_START_DELAY:
                    cv2.putText(frame, "DRAWING", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if draw_counter == DRAW_START_DELAY:
                        prev_x, prev_y = x, y
                        continue

                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y),
                                   brush_color, brush_thickness)

                    prev_x, prev_y = x, y
                else:
                    cv2.putText(frame, "READY...", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    prev_x, prev_y = None, None

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
        lost_frames += 1

        if lost_frames > LOST_FRAME_LIMIT:
            draw_counter = 0
            prev_x, prev_y = None, None
            points_buffer.clear()

    output = cv2.addWeighted(frame, 0.7, canvas, 0.7, 0)

    cv2.imshow("Air Drawing", output)
    cv2.imshow("Canvas Only", canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        canvas = np.zeros_like(frame)
        prev_x, prev_y = None, None
        points_buffer.clear()
        draw_counter = 0
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