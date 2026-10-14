"""قراءة النص من اللوحة باستخدام Tesseract."""
import platform
import re

import numpy as np
import pytesseract

from .preprocessing import build_variants

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def clean_text(text):
    text = text.replace("\n", " ").strip()
    text = re.sub(r"[^0-9A-Za-z\u0600-\u06FF\- ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def available_languages():
    try:
        return pytesseract.get_languages(config="")
    except Exception:
        return ["eng"]


def run_ocr(img, psm=7):
    langs = available_languages()
    lang = "eng+ara" if "ara" in langs else "eng"
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        img, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    words, confs = [], []
    for t, c in zip(data["text"], data["conf"]):
        t = t.strip()
        try:
            c = float(c)
        except ValueError:
            continue
        if t and c > 0:
            words.append(t)
            confs.append(c)
    return clean_text(" ".join(words)), (float(np.mean(confs)) if confs else 0.0)


def read_plate(plate_bgr):
    """يجرب كل طرق المعالجة ويرجع (أفضل نتيجة، كل المحاولات)."""
    attempts = []
    for name, img in build_variants(plate_bgr).items():
        for psm in (7, 8):
            try:
                text, conf = run_ocr(img, psm)
            except Exception:
                continue
            attempts.append({"method": f"{name} (psm {psm})", "text": text,
                             "conf": conf, "image": img})

    valid = [a for a in attempts if any(ch.isdigit() for ch in a["text"])]
    best = max(valid, key=lambda a: a["conf"] + min(len(a["text"]), 10), default=None)
    return best, attempts
