import platform
import re
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract
OCR\tesseract.exe"
st.set_page_config(page_title="ﻛﺸﻒ ﻟﻮﺣﺎت اﻟﺴﻴﺎرات", page_icon="
🚗
", 
layout="centered")============ ﻣﻌﺎﻟﺠﺔ اﻟﺼﻮرة ============ #

def resize_keep_ratio(img, max_width=1200):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    s = max_width / w
    return cv2.resize(img, (int(w * s), int(h * s)), 
interpolation=cv2.INTER_AREA)
def unsharp(gray, amount=1.5, sigma=3):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)
def deskew(img):
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
    plate = cv2.resize(plate_bgr, None, fx=3, fy=3, 
interpolation=cv2.INTER_CUBIC)
    gray = deskew(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = unsharp(clahe.apply(gray))
    denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + 
cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(enhanced, 255, 
cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return {,enhanced :"ﺗﺒﺎﻳﻦ + ﺣﺪة"       
 
        "Otsu": otsu,,adaptive :"ﻋﺘﺒﺔ ﺗﻜﻴﻔﻴﺔ"       
 ,cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel) :"إﺻﻼح اﻟﻄﻤﺲ"       
 ,cv2.medianBlur(otsu, 3) :"إزاﻟﺔ اﻟﺒﻘﻊ"       
 ,cv2.bitwise_not(otsu) :"ﻣﻌﻜﻮﺳﺔ"       
 
    }============ اﻛﺘﺸﺎف اﻟﻠﻮﺣﺔ ============ #

def find_plate(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, 
cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]
    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.004 or area > img_area * 0.5:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (1.8 <= w / float(h) <= 6.5):
            continue
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        score = area * (1.5 if len(approx) == 4 else 1.0)
        if y + h / 2 > img_h * 0.45:
            score *= 1.3
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))
    annotated = img_bgr.copy()
    if best is None:
        return None, annotated
    x, y, w, h = best[1]
    pad = int(0.05 * h)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, img_w), min(y + h + pad, img_h)
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 3)
    return img_bgr[y0:y1, x0:x1], annotated
# ============ OCR ============
def clean_text(text):
    text = re.sub(r"[^0-9A-Za-z\u0600-\u06FF\- ]", "", text.replace("\n", " 
"))
    return re.sub(r"\s+", " ", text).strip()
@st.cache_data
def available_languages():
    try:
        return pytesseract.get_languages(config="")
    except Exception:
        return ["eng"]
def run_ocr(img, psm=7):
    lang = "eng+ara" if "ara" in available_languages() else "eng"
    data = pytesseract.image_to_data(img, lang=lang, config=f"--oem 3 --psm 
{psm}",
                                     output_type=pytesseract.Output.DICT)
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
    return clean_text(" ".join(words)), (float(np.mean(confs)) if confs else 
0.0)
def read_plate(plate_bgr):
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
    best = max(valid, key=lambda a: a["conf"] + min(len(a["text"]), 10), 
default=None)
    return best, attempts============ ﺗﺤﻠﻴﻞ اﻟﻠﻮﺣﺔ اﻟﻌﺮاﻗﻴﺔ ============ #

_AR = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PATTERN = re.compile(r"(\d{1,2})\s*-?\s*([A-Za-z\u0600-\u06FF])\s*-?\s*
(\d{3,6})")
def parse_plate(text):
    m = PATTERN.search((text or "").translate(_AR))
    if not m:
        return None
    code, letter, number = m.groups()
    return {"code": code, "letter": letter.upper(), "number": number,
            "formatted": f"{code}-{letter.upper()} {number}"}============ اﻟﻮاﺟﻬﺔ ============ #

if "history" not in st.session_state:
    st.session_state.history = []
st.markdown("<h2 style='text-align:center'>
🚗,">h2/<ﻛﺸﻒ واﺳﺘﺨﺮاج رﻗﻢ ﻟﻮﺣﺔ اﻟﺴﻴﺎرة
 
            unsafe_allow_html=True)
st.caption("ﻳﻌﻤﻞ ﻣﻊ اﻟﺼﻮر ﻏﻴﺮ اﻟﻮاﺿﺤﺔ أو اﻟﻤﻄﻤﻮﺳﺔ أو اﻟﺘﻲ ﻋﻠﻴﻬﺎ ﻃﻴﻦ.")
with st.sidebar:
    st.header("اﻹﻋﺪادات")
    show_debug = st.checkbox("ﻋﺮض ﺗﻔﺎﺻﻴﻞ اﻟﻤﻌﺎﻟﺠﺔ", value=False)
    langs = available_languages()
    st.write("ﻟﻐﺎت OCR اﻟﻤﺘﻮﻓﺮة:", ", ".join(langs))
    if "ara" not in langs:
        st.warning("ﺣﺰﻣﺔ اﻟﻌﺮﺑﻴﺔ ﻏﻴﺮ ﻣﺜﺒﺘﺔ. ﺗﺄﻛﺪ ﻣﻦ packages.txt")
source = st.radio("["رﻓﻊ ﺻﻮرة", "اﻟﻜﺎﻣﻴﺮا"] ,"ﻣﺼﺪر اﻟﺼﻮرة, horizontal=True)
file = (st.file_uploader("ارﻓﻊ ﺻﻮرة اﻟﺴﻴﺎرة", type=["jpg", "jpeg", "png"])
        if source == "رﻓﻊ ﺻﻮرة" else st.camera_input("اﻟﺘﻘﻂ ﺻﻮرة"))
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="اﻟﺼﻮرة اﻟﻤﺮﻓﻮﻋﺔ", use_container_width=True)
    if st.button("ﺑﺪء ﻋﻤﻠﻴﺔ اﻟﻔﺤﺺ واﺳﺘﺨﺮاج رﻗﻢ اﻟﻠﻮﺣﺔ", type="primary"):
        try:
            with st.spinner("ﺟﺎري اﻟﻤﻌﺎﻟﺠﺔ..."):
                img_bgr = resize_keep_ratio(cv2.cvtColor(np.array(image), 
cv2.COLOR_RGB2BGR))
                plate, annotated = find_plate(img_bgr)
                if plate is None:
                    st.warning("ﻟﻢ ﻳﺘﻢ اﻟﻌﺜﻮر ﻋﻠﻰ ﻟﻮﺣﺔ، ﺳﻨﻘﺮأ اﻟﺼﻮرة ﻛﺎﻣﻠﺔ.")
                    plate = img_bgr
                else:
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             caption="ﻣﻮﻗﻊ اﻟﻠﻮﺣﺔ اﻟﻤﻜﺘﺸﻒ", 
use_container_width=True)
                best, attempts = read_plate(plate)
            st.image(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), caption="اﻟﻠﻮﺣﺔ )"اﻟﻤﻘﺼﻮﺻ
ﺔ
            if best:
                parsed = parse_plate(best["text"])
                final = parsed["formatted"] if parsed else best["text"]
                st.success(f"رﻗﻢ اﻟﻠﻮﺣﺔ: {final}")
                c1, c2 = st.columns(2)
                c1.metric("ﻧﺴﺒﺔ اﻟﺜﻘﺔ", f"{best['conf']:.0f}%")
                c2.metric("ﻃﺮﻳﻘﺔ اﻟﻤﻌﺎﻟﺠﺔ", best["method"].split(" (")[0])
                if parsed:
                    st.write(f"رﻣﺰ اﻟﻤﺤﺎﻓﻈﺔ: **{parsed['code']}** | اﻟﺤﺮف: **
{parsed['letter']}** | اﻟﺮﻗﻢ: **{parsed['number']}**")
                if best["conf"] < 50:
                    st.info("اﻟﺜﻘﺔ ﻣﻨﺨﻔﻀﺔ. ﺟﺮّب ﺻﻮرة أوﺿﺢ.")
                st.image(best["image"], caption="اﻟﻠﻮﺣﺔ ﺑﻌﺪ اﻟﺘﺤﺴﻴﻦ", clamp=True)
                st.session_state.history.append({,datetime.now().strftime("%Y-%m-%d %H:%M:%S") :"اﻟﻮﻗﺖ"                   
 )}round(best["conf"]) :"% اﻟﺜﻘﺔ" ,final :"اﻟﺮﻗﻢ"                   
 
            else:
                st.error("ﺗﻌﺬ ّ ر اﺳﺘﺨﺮاج اﻟﺮﻗﻢ. ﺟﺮّب ﺻﻮرة أوﺿﺢ.")
            if show_debug:
                st.subheader("ﻛﻞ اﻟﻤﺤﺎوﻻت")
                for a in attempts:
                    st.write(f"{a['method']} → `{a['text']}` 
({a['conf']:.0f}%)")
                    st.image(a["image"], clamp=True)
        except Exception as e:
            st.error(f"ﺣﺪث ﺧﻄﺄ أﺛﻨﺎء اﻟﻤﻌﺎﻟﺠﺔ: {e}")
if st.session_state.history:
    st.divider()
    st.subheader("ﺳﺠﻞ اﻟﻨﺘﺎﺋﺞ")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
    st.download_button("ﺗﻨﺰﻳﻞ اﻟﺴﺠﻞ CSV", df.to_csv(index=False).encode("utf-8
sig"),
                       "plates.csv", "text/csv")
