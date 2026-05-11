import cv2
import mediapipe as mp
import numpy as np
from collections import deque

points_buffer = deque(maxlen=5)
trajectory_points = []
is_drawing_shape = False

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

canvas = None
canvas_history = []
prev_x, prev_y = None, None
draw_counter = 0
brush_color = (255, 0, 0)  # blue
brush_thickness = 5
eraser_thickness = 20
lost_frames = 0
last_velocity = (0, 0)
MAX_PREDICT_FRAMES = 5
LOST_FRAME_LIMIT = 8
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

def calculateAngle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle

def isClosedStroke(trajectory_points, threshold=60):
    if len(trajectory_points) < 2:
        return False

    start = np.array(trajectory_points[0])
    end = np.array(trajectory_points[-1])

    distance = np.linalg.norm(start - end)

    pts = np.array(trajectory_points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    bbox_diagonal = np.sqrt(w ** 2 + h ** 2)

    return distance < threshold or distance < 0.25 * bbox_diagonal


def isRectangleLike(approx):
    if len(approx) != 4:
        return False

    points = [tuple(p[0]) for p in approx]

    angles = []
    for i in range(4):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[(i + 1) % 4]
        angles.append(calculateAngle(p1, p2, p3))

    return all(60 <= angle <= 120 for angle in angles)

def removeRoughShape(canvas, trajectory_points, thickness):
    if len(trajectory_points) < 2:
        return canvas

    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    pts = np.array(trajectory_points, dtype=np.int32)

    cv2.polylines(mask, [pts], False, 255, thickness + 4)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=0)

    canvas[mask == 255] = 0
    return canvas

def refineShape(canvas, trajectory_points, color, thickness):

    if len(trajectory_points) < 15:
        print("Not enough points")
        return canvas

    pts = np.array(trajectory_points, dtype=np.int32)

    # Create binary image from trajectory
    temp = np.zeros(canvas.shape[:2], dtype=np.uint8)

    cv2.polylines(
        temp,
        [pts],
        False,
        255,
        thickness
    )

    # Morphological closing to connect gaps
    kernel = np.ones((5, 5), np.uint8)

    temp = cv2.morphologyEx(
        temp,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Edge detection
    edges = cv2.Canny(temp, 50, 150)

    # -----------------------------
    # 1. LINE DETECTION (HOUGH)
    # -----------------------------
    closed_stroke = isClosedStroke(trajectory_points)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=25,
        maxLineGap=80
    )

    if lines is not None and not closed_stroke:
        longest = max(
            lines,
            key=lambda l: np.linalg.norm(
                (l[0][0] - l[0][2], l[0][1] - l[0][3])
            )
        )

        x1, y1, x2, y2 = longest[0]

        line_length = np.linalg.norm((x1 - x2, y1 - y2))

        x, y, w, h = cv2.boundingRect(pts)

        bbox_diagonal = np.sqrt(w ** 2 + h ** 2)

        # More tolerant line rule
        if line_length > 0.55 * bbox_diagonal:
            canvas = removeRoughShape(canvas, trajectory_points, thickness)

            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)

            print("Detected: LINE")
            return canvas

    # -----------------------------
    # CONTOUR DETECTION
    # -----------------------------

    contours, _ = cv2.findContours(
        temp,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("No contours")
        return canvas

    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)

    if area < 300:
        print("Too small")
        return canvas

    perimeter = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(
        contour,
        0.04 * perimeter,
        True
    )

    # -----------------------------
    # 2. TRIANGLE
    # -----------------------------

    if len(approx) == 3:

        canvas = removeRoughShape(
            canvas,
            trajectory_points,
            thickness
        )

        cv2.drawContours(
            canvas,
            [approx],
            -1,
            color,
            thickness
        )

        print("Detected: TRIANGLE")
        return canvas

    # -----------------------------
    # 3. RECTANGLE / SQUARE
    # -----------------------------

    if len(approx) == 4:

        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect)

        box = np.int32(box)

        w_rect, h_rect = rect[1]

        if w_rect == 0 or h_rect == 0:
            return canvas

        ratio = max(w_rect, h_rect) / min(w_rect, h_rect)

        if ratio < 1.2:
            shape_name = "SQUARE"
        else:
            shape_name = "RECTANGLE"

        canvas = removeRoughShape(
            canvas,
            trajectory_points,
            thickness
        )

        cv2.drawContours(
            canvas,
            [box],
            0,
            color,
            thickness
        )

        print("Detected:", shape_name)
        return canvas

    # -----------------------------
    # 4. CIRCLE (HOUGH)
    # -----------------------------

    circles = cv2.HoughCircles(
        temp,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=25,
        minRadius=20,
        maxRadius=300
    )

    if circles is not None:

        circles = np.uint16(np.around(circles))

        c = circles[0][0]

        center = (c[0], c[1])

        radius = c[2]

        canvas = removeRoughShape(
            canvas,
            trajectory_points,
            thickness
        )

        cv2.circle(
            canvas,
            center,
            radius,
            color,
            thickness
        )

        print("Detected: CIRCLE")
        return canvas

    print("No reliable shape detected")

    return canvas


def saveCanvasState(canvas, canvas_history, max_history=20):
    if canvas is not None:
        canvas_history.append(canvas.copy())

    if len(canvas_history) > max_history:
        canvas_history.pop(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # Better for MediaPipe: do NOT blur before hand tracking
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
                        saveCanvasState(canvas, canvas_history)
                        trajectory_points.clear()
                        trajectory_points.append((x, y))
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
                        saveCanvasState(canvas, canvas_history)
                        trajectory_points.clear()
                        trajectory_points.append((x, y))
                        prev_x, prev_y = x, y
                        continue

                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y),
                                 brush_color, brush_thickness)

                        last_velocity = (x - prev_x, y - prev_y)

                    trajectory_points.append((x, y))
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

    cv2.imshow("Air Drawing", output)
    cv2.imshow("Canvas Only", canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        canvas = np.zeros_like(frame)
        prev_x, prev_y = None, None
        points_buffer.clear()
        trajectory_points.clear()
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

    if key == ord("s"):
        saveCanvasState(canvas, canvas_history)
        canvas = refineShape(canvas, trajectory_points, brush_color, brush_thickness)
        trajectory_points.clear()
        prev_x, prev_y = None, None
        points_buffer.clear()
        draw_counter = 0
    
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