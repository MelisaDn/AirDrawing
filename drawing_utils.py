def smoothPoint(x, y, points_buffer):
    """Smooth fingertip coordinates using a moving average over recent points."""
    points_buffer.append((x, y))

    avg_x = int(sum(p[0] for p in points_buffer) / len(points_buffer))
    avg_y = int(sum(p[1] for p in points_buffer) / len(points_buffer))

    return avg_x, avg_y


def saveCanvasState(canvas, canvas_history, max_history=20):
    """Save a copy of the current canvas so undo can restore it later."""
    if canvas is not None:
        canvas_history.append(canvas.copy())

    if len(canvas_history) > max_history:
        canvas_history.pop(0)
