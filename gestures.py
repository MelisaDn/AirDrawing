def distance(p1, p2):
    """Calculate normalized Euclidean distance between two MediaPipe landmarks."""
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


def isFingerExtended(lm, tip_id, pip_id, mcp_id):
    """Check whether one finger is extended based on tip/PIP/MCP distance from wrist."""
    tip_to_wrist = distance(lm[tip_id], lm[0])
    pip_to_wrist = distance(lm[pip_id], lm[0])
    mcp_to_wrist = distance(lm[mcp_id], lm[0])

    return tip_to_wrist > pip_to_wrist and tip_to_wrist > mcp_to_wrist


def isOnlyIndexFingerUp(hand_landmarks):
    """Return True when only the index finger is extended; used for drawing."""
    lm = hand_landmarks.landmark

    index_extended = isFingerExtended(lm, 8, 6, 5)
    middle_extended = isFingerExtended(lm, 12, 10, 9)
    ring_extended = isFingerExtended(lm, 16, 14, 13)
    pinky_extended = isFingerExtended(lm, 20, 18, 17)

    return index_extended and not middle_extended and not ring_extended and not pinky_extended


def isPeaceSign(hand_landmarks):
    """Return True for index + middle finger up; used to pause drawing."""
    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y

    return index_up and middle_up and ring_down and pinky_down


def isOpenHand(hand_landmarks):
    """Return True when four fingers are up; used as the eraser gesture."""
    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    return index_up and middle_up and ring_up and pinky_up
