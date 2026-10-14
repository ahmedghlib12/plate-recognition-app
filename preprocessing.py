"""معالجة الصور: تحسين الصور الضبابية والمطينة والمطموسة."""
import cv2
import numpy as np


def resize_keep_ratio(img, max_width=1200):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def unsharp(gray, amount=1.5, sigma=3):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def deskew(gray_or_bin):
    """تعديل ميلان اللوحة البسيط."""
    img = gray_or_bin
    inv = cv2.bitwise_not(img) if img.mean() > 127 else img
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) < 50:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > 15 or abs(angle) < 0.5:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def build_variants(plate_bgr):
    """يرجع قاموساً: اسم الطريقة -> صورة محسّنة جاهزة للـ OCR."""
    plate = cv2.resize(plate_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = deskew(gray)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = unsharp(clahe.apply(gray))

    denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    repaired = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.medianBlur(otsu, 3)

    return {
        "تباين + حدة": enhanced,
        "Otsu": otsu,
        "عتبة تكيفية": adaptive,
        "إصلاح الطمس": repaired,
        "إزالة البقع": cleaned,
        "معكوسة": cv2.bitwise_not(otsu),
    }
