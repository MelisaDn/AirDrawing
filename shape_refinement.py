import cv2
import numpy as np


def calculateAngle(p1, p2, p3):
    """Calculate the angle at p2 using three contour points."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle


def isClosedStroke(trajectory_points, threshold=60):
    """Check whether the drawn stroke starts and ends close enough to be a closed shape."""
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
    """Check whether a four-point contour has roughly rectangular angles."""
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
    """Remove the rough user-drawn stroke before drawing the refined shape."""
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
    """Detect and redraw a cleaner line, triangle, rectangle/square, or circle."""

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
