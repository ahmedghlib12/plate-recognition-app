"""اكتشاف مكان لوحة السيارة داخل الصورة."""
import cv2
import numpy as np


def find_plate(img_bgr):
    """يرجع (صورة اللوحة أو None، الصورة مع المربع، الإحداثيات أو None)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]

    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w
    best = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.004 or area > img_area * 0.5:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / float(h)
        if not (1.8 <= ratio <= 6.5):
            continue
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        score = area * (1.5 if len(approx) == 4 else 1.0)
        if y + h / 2 > img_h * 0.45:  # اللوحة غالباً في النصف السفلي
            score *= 1.3
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))

    annotated = img_bgr.copy()
    if best is None:
        return None, annotated, None

    x, y, w, h = best[1]
    pad = int(0.05 * h)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, img_w), min(y + h + pad, img_h)
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 3)
    return img_bgr[y0:y1, x0:x1], annotated, (x0, y0, x1, y1)
