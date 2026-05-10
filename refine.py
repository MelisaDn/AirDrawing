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

def calculateAngle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle


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

    cv2.polylines(mask, [pts], False, 255, thickness + 12)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    canvas[mask == 255] = 0
    return canvas

def resamplePoints(points, n=100):
    """Resample trajectory to n evenly spaced points for consistent analysis."""
    if len(points) < 2:
        return points
    pts = np.array(points, dtype=np.float32)
    dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    cumlen = np.concatenate([[0], np.cumsum(dists)])
    total = cumlen[-1]
    if total == 0:
        return points
    target = np.linspace(0, total, n)
    xs = np.interp(target, cumlen, pts[:, 0])
    ys = np.interp(target, cumlen, pts[:, 1])
    return list(zip(xs.astype(int), ys.astype(int)))


def isClosedShape(trajectory_points, threshold=0.2):
    """Check if start and end points are close relative to shape size."""
    if len(trajectory_points) < 10:
        return False
    start = np.array(trajectory_points[0])
    end   = np.array(trajectory_points[-1])
    pts   = np.array(trajectory_points)
    bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    if bbox_diag == 0:
        return False
    return np.linalg.norm(end - start) / bbox_diag < threshold


def refineShape(canvas, trajectory_points, color, thickness):
    if len(trajectory_points) < 15:
        print("Not enough points")
        return canvas

    # --- 1. Resample for consistent analysis ---
    sampled = resamplePoints(trajectory_points, n=120)
    pts_raw = np.array(trajectory_points, dtype=np.int32)
    pts     = np.array(sampled, dtype=np.int32)

    closed = isClosedShape(trajectory_points)

    # --- 2. Build a clean binary mask from trajectory ---
    temp = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.polylines(temp, [pts_raw], False, 255, thickness + 4)
    kernel = np.ones((7, 7), np.uint8)
    temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(temp, 50, 150)

    # -----------------------------------------------
    # A. OPEN SHAPES  →  LINE  /  ARROW
    # -----------------------------------------------
    if not closed:

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=30, minLineLength=60, maxLineGap=30
        )

        if lines is not None:
            longest = max(lines,
                          key=lambda l: np.linalg.norm((l[0][0]-l[0][2],
                                                        l[0][1]-l[0][3])))
            x1, y1, x2, y2 = longest[0]
            line_len     = np.linalg.norm((x1-x2, y1-y2))
            contour_len  = cv2.arcLength(pts_raw, False)

            if line_len > 0.55 * contour_len:
                canvas = removeRoughShape(canvas, trajectory_points, thickness)
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
                print("Detected: LINE")
                return canvas

        print("No reliable open shape detected")
        return canvas

    # -----------------------------------------------
    # B. CLOSED SHAPES  →  CIRCLE / ELLIPSE / POLYGON
    # -----------------------------------------------

    contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found")
        return canvas

    contour = max(contours, key=cv2.contourArea)
    area     = cv2.contourArea(contour)

    if area < 300:
        print("Shape too small")
        return canvas

    perimeter = cv2.arcLength(contour, True)

    # --- B1. CIRCLE check via circularity ---
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

    if circularity > 0.72:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        canvas = removeRoughShape(canvas, trajectory_points, thickness)
        cv2.circle(canvas, (int(cx), int(cy)), int(radius), color, thickness)
        print(f"Detected: CIRCLE  (circularity={circularity:.2f})")
        return canvas

    # --- B2. ELLIPSE check ---
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (ex, ey), (major, minor), angle = ellipse
        if minor > 0:
            ellipse_ratio = major / minor
            # Accept elongated ellipses but reject near-circles (already handled)
            # and near-lines
            if 1.3 < ellipse_ratio < 5.0:
                # Verify fit quality: compare fitted ellipse area vs contour area
                ellipse_area = np.pi * (major/2) * (minor/2)
                fit_quality  = min(area, ellipse_area) / max(area, ellipse_area)
                if fit_quality > 0.65:
                    canvas = removeRoughShape(canvas, trajectory_points, thickness)
                    cv2.ellipse(canvas, ellipse, color, thickness)
                    print(f"Detected: ELLIPSE  (ratio={ellipse_ratio:.2f}, fit={fit_quality:.2f})")
                    return canvas

    # --- B3. POLYGON approximation ---
    epsilon  = 0.03 * perimeter
    approx   = cv2.approxPolyDP(contour, epsilon, True)
    n_sides  = len(approx)

    # Retry with looser epsilon if we got too many sides
    if n_sides > 6:
        epsilon = 0.05 * perimeter
        approx  = cv2.approxPolyDP(contour, epsilon, True)
        n_sides = len(approx)

    # Triangle
    if n_sides == 3:
        canvas = removeRoughShape(canvas, trajectory_points, thickness)
        cv2.drawContours(canvas, [approx], -1, color, thickness)
        print("Detected: TRIANGLE")
        return canvas

    # Rectangle / Square  — with angle quality check
    if n_sides == 4:
        rect  = cv2.minAreaRect(contour)
        box   = np.int32(cv2.boxPoints(rect))
        w_r, h_r = rect[1]

        if w_r > 0 and h_r > 0:
            ratio      = max(w_r, h_r) / min(w_r, h_r)
            shape_name = "SQUARE" if ratio < 1.25 else "RECTANGLE"

            # Angle quality: all interior angles should be ≈90°
            corners = [tuple(p[0]) for p in approx]
            angles  = [
                calculateAngle(corners[i-1], corners[i], corners[(i+1) % 4])
                for i in range(4)
            ]
            if all(65 <= a <= 115 for a in angles):
                canvas = removeRoughShape(canvas, trajectory_points, thickness)
                cv2.drawContours(canvas, [box], 0, color, thickness)
                print(f"Detected: {shape_name}  angles={[f'{a:.0f}' for a in angles]}")
                return canvas

    # Pentagon / Hexagon
    if n_sides in (5, 6):
        canvas = removeRoughShape(canvas, trajectory_points, thickness)
        cv2.drawContours(canvas, [approx], -1, color, thickness)
        print(f"Detected: {n_sides}-SIDED POLYGON")
        return canvas

    print(f"No reliable shape detected  (sides={n_sides}, circularity={circularity:.2f})")
    return canvas

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
        canvas = refineShape(canvas, trajectory_points, brush_color, brush_thickness)
        trajectory_points.clear()
        prev_x, prev_y = None, None
        points_buffer.clear()
        draw_counter = 0

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