"""
نظام التعرف على أرقام لوحات السيارات في الظروف الصعبة
(تشوه المنظور، ضبابية، ضعف إضاءة)

خط المعالجة:
1) تحسين الإضاءة العام للصورة
2) كشف اللوحة بطريقتين (الحواف + المورفولوجيا) مع ترتيب المرشحين
3) تصحيح المنظور (Perspective) وقص اللوحة
4) تحسين اللوحة: تكبير، إزالة ضوضاء، تصحيح إضاءة، CLAHE، شحذ، تعديل ميلان
5) توليد عدة نسخ معالجة من اللوحة
6) قراءة كل نسخة بعدة أوضاع (Tesseract + EasyOCR اختياري)
7) تصويت على مستوى كل حرف للوصول لأدق نتيجة
"""

import platform
import re
import shutil
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, ImageOps

# ----------------------------------------------------------------------
# إعداد Tesseract
# ----------------------------------------------------------------------
if platform.system() == "Windows":
    _win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = _win_path
# على لينكس (Streamlit Cloud) يتم إيجاده تلقائياً من PATH

try:
    from plate_detector import detect_with_yolo, model_available as yolo_available
except Exception:  # onnxruntime غير مثبت
    def yolo_available():
        return False

try:
    import easyocr  # اختياري: شبكة عصبية عميقة

    HAS_EASYOCR = True
except Exception:
    HAS_EASYOCR = False

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
EN_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
            "Z": "2", "S": "5", "G": "6", "B": "8", "T": "7"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}


# ----------------------------------------------------------------------
# أدوات عامة
# ----------------------------------------------------------------------
def load_image(file, max_side=1600):
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        arr = cv2.resize(arr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return arr


def to_rgb(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def auto_gamma(gray):
    """رفع سطوع الصور المظلمة تلقائياً."""
    m = float(np.clip(gray.mean() / 255.0, 0.05, 0.95))
    exp = float(np.clip(np.log(0.5) / np.log(m), 0.4, 1.5))
    table = (np.power(np.arange(256) / 255.0, exp) * 255).astype("uint8")
    return cv2.LUT(gray, table)


def global_enhance(img):
    """تحسين الإضاءة قبل الكشف (قناة L فقط)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = auto_gamma(l)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------------------
# كشف اللوحة
# ----------------------------------------------------------------------
def _box_score(box, shape, fill=1.0):
    h, w = shape[:2]
    (_, _), (cw, ch), _ = cv2.minAreaRect(box.astype("float32"))
    if cw == 0 or ch == 0:
        return 0.0
    aspect = max(cw, ch) / min(cw, ch)
    area_ratio = (cw * ch) / (w * h)
    if not (1.8 <= aspect <= 7.0) or not (0.003 <= area_ratio <= 0.5):
        return 0.0
    aspect_score = 1 - min(abs(aspect - 4.0) / 4.0, 1)
    return float((0.5 + aspect_score) * np.sqrt(area_ratio) * fill)


def detect_plates(img, top_n=3):
    """يعيد قائمة بأفضل مرشحي اللوحة: [(score, box4x2)]."""
    enh = global_enhance(img)
    gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cands = []

    # الطريقة A: الحواف + مضلعات رباعية
    g = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.dilate(cv2.Canny(g, 30, 200), None, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:50]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            box = approx.reshape(4, 2).astype("float32")
            (_, _), (cw, ch), _ = cv2.minAreaRect(box)
            fill = cv2.contourArea(approx) / max(cw * ch, 1)
            s = _box_score(box, img.shape, fill)
            if s > 0:
                cands.append((s, box))

    # الطريقة B: Blackhat + Sobel + مناطق ساطعة (تتحمل التشوه والضبابية)
    kw = max(9, w // 40)
    rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, max(3, kw // 3)))
    sq_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_k)
    grad = np.absolute(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1))
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    grad = cv2.GaussianBlur(grad, (5, 5), 0)
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect_k)
    thr = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, sq_k)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thr = cv2.bitwise_and(thr, thr, mask=light)
    thr = cv2.dilate(cv2.erode(thr, None, iterations=2), None, iterations=3)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:30]:
        box = cv2.boxPoints(cv2.minAreaRect(c)).astype("float32")
        s = _box_score(box, img.shape, 0.9)
        if s > 0:
            cands.append((s, box))

    # إزالة التكرار (مراكز متقاربة)
    cands.sort(key=lambda x: x[0], reverse=True)
    kept = []
    for s, box in cands:
        cen = box.mean(axis=0)
        size = min(np.ptp(box[:, 0]), np.ptp(box[:, 1]))
        if all(np.linalg.norm(cen - k[1].mean(axis=0)) > 0.5 * max(size, 1)
               for k in kept):
            kept.append((s, box))
        if len(kept) >= top_n:
            break
    return kept


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect


def warp_plate(img, box, pad=0.06):
    """تصحيح المنظور وقص اللوحة."""
    rect = order_points(box)
    cen = rect.mean(axis=0)
    rect = cen + (rect - cen) * (1 + pad)
    tl, tr, br, bl = rect
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if w < h:  # لوحة ملتقطة بشكل عمودي
        rect = np.array([bl, tl, tr, br], dtype="float32")
        w, h = h, w
    w, h = max(w, 20), max(h, 10)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------------------
# تحسين اللوحة
# ----------------------------------------------------------------------
def deskew(gray):
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    pts = cv2.findNonZero(th)
    if pts is None or len(pts) < 50:
        return gray
    angle = cv2.minAreaRect(pts)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance_plate(plate, o):
    h, w = plate.shape[:2]
    scale = max(1.0, 500.0 / w, 100.0 / h)
    big = cv2.resize(plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    if o["denoise"]:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10,
                                        templateWindowSize=7, searchWindowSize=21)
    if o["lowlight"]:
        gray = auto_gamma(gray)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if o["sharpen"]:
        amount = 1.8 if blur_var < 100 else 0.8  # ضبابية أكثر = شحذ أقوى
        soft = cv2.GaussianBlur(gray, (0, 0), 3)
        gray = cv2.addWeighted(gray, 1 + amount, soft, -amount, 0)
    if o["deskew"]:
        gray = deskew(gray)
    return big, gray, blur_var


def _white_bg(binary):
    return binary if (binary > 127).mean() >= 0.5 else cv2.bitwise_not(binary)


def make_variants(gray):
    """عدة نسخ من اللوحة لزيادة فرص القراءة الصحيحة."""
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    otsu = _white_bg(otsu)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    adaptive = _white_bg(adaptive)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    morph = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, k)
    variants = {
        "gray": gray,
        "otsu": otsu,
        "otsu_inverted": cv2.bitwise_not(otsu),
        "adaptive": adaptive,
        "morph_clean": morph,
    }
    out = {}
    for name, im in variants.items():
        val = int(np.median(im)) if name == "gray" else 255
        if name == "otsu_inverted":
            val = 0
        out[name] = cv2.copyMakeBorder(im, 15, 15, 15, 15,
                                       cv2.BORDER_CONSTANT, value=val)
    return out


# ----------------------------------------------------------------------
# محركات القراءة
# ----------------------------------------------------------------------
def normalize_text(t):
    t = t.translate(ARABIC_DIGITS)
    return re.sub(r"[^0-9A-Za-z\u0621-\u064A]", "", t).upper()


def run_tesseract(img, lang, psm, whitelist):
    cfg = f"--oem 3 --psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    d = pytesseract.image_to_data(img, lang=lang, config=cfg,
                                  output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for t, c in zip(d["text"], d["conf"]):
        try:
            c = float(c)
        except (TypeError, ValueError):
            continue
        if t.strip() and c >= 0:
            words.append(t.strip())
            confs.append(c)
    return "".join(words), (sum(confs) / len(confs) if confs else 0.0)


if HAS_EASYOCR:

    @st.cache_resource(show_spinner="جاري تحميل نموذج EasyOCR (أول مرة فقط)...")
    def load_easyocr(langs):
        return easyocr.Reader(list(langs), gpu=False)


def run_easyocr(img, langs, english_only):
    reader = load_easyocr(tuple(langs))
    kw = {"allowlist": EN_WHITELIST} if english_only else {}
    res = reader.readtext(img, detail=1, paragraph=False, **kw)
    if not res:
        return "", 0.0
    res.sort(key=lambda r: r[0][0][0])  # ترتيب من اليسار لليمين
    text = "".join(r[1] for r in res)
    conf = 100 * sum(r[2] for r in res) / len(res)
    return text, conf


def add_candidate(cands, engine, variant, psm, raw, conf):
    text = normalize_text(raw)
    if len(text) >= 3:
        factor = 1.0 if 4 <= len(text) <= 10 else 0.6
        cands.append({"engine": engine, "variant": variant, "psm": psm,
                      "raw": raw.strip(), "text": text,
                      "conf": round(conf, 1), "score": conf * factor})


def read_plate(plate, o):
    big, gray, blur_var = enhance_plate(plate, o)
    variants = make_variants(gray)
    cands = []

    if "Tesseract" in o["engines"]:
        lang = "+".join(o["langs"])
        wl = EN_WHITELIST if o["langs"] == ["eng"] else None
        for name, im in variants.items():
            for psm in (7, 8, 13):
                try:
                    raw, conf = run_tesseract(im, lang, psm, wl)
                except pytesseract.TesseractError:
                    continue
                add_candidate(cands, "Tesseract", name, psm, raw, conf)

    if "EasyOCR" in o["engines"] and HAS_EASYOCR:
        ez_langs = [{"eng": "en", "ara": "ar"}[l] for l in o["langs"]
                    if l in ("eng", "ara")] or ["en"]
        eng_only = ez_langs == ["en"]
        for name, im in (("gray", variants["gray"]), ("otsu", variants["otsu"]),
                         ("color_upscaled", big)):
            try:
                raw, conf = run_easyocr(im, ez_langs, eng_only)
            except Exception:
                continue
            add_candidate(cands, "EasyOCR", name, "-", raw, conf)

    return {"enhanced": gray, "variants": variants, "cands": cands,
            "blur": blur_var}


# ----------------------------------------------------------------------
# التصويت والتصحيح
# ----------------------------------------------------------------------
def consensus(cands):
    """تصويت على مستوى كل حرف بين كل القراءات ذات الطول الأكثر شيوعاً."""
    if not cands:
        return "", 0.0, 0, 0
    by_len = defaultdict(float)
    for c in cands:
        by_len[len(c["text"])] += c["score"]
    length = max(by_len, key=by_len.get)
    same = [c for c in cands if len(c["text"]) == length]
    out = []
    for i in range(length):
        votes = defaultdict(float)
        for c in same:
            votes[c["text"][i]] += c["score"]
        out.append(max(votes, key=votes.get))
    text = "".join(out)
    agree = sum(1 for c in same if c["text"] == text)
    conf = sum(c["conf"] for c in same) / len(same)
    return text, conf, agree, len(same)


def apply_pattern(text, pattern):
    """تصحيح الالتباس (O/0, I/1, B/8 ...) حسب نمط اللوحة: D رقم، L حرف."""
    pattern = pattern.strip().upper()
    if not pattern or len(pattern) != len(text):
        return text
    out = []
    for ch, p in zip(text, pattern):
        if p == "D":
            ch = TO_DIGIT.get(ch, ch)
        elif p == "L":
            ch = TO_LETTER.get(ch, ch)
        out.append(ch)
    return "".join(out)


# ----------------------------------------------------------------------
# الواجهة
# ----------------------------------------------------------------------
st.set_page_config(page_title="التعرف على لوحات السيارات", page_icon="🚗",
                   layout="wide")
st.title("🚗 التعرف على أرقام لوحات السيارات في الظروف الصعبة")
st.caption("يعالج التشوه والضبابية وضعف الإضاءة عبر: كشف ذكي، تصحيح منظور، "
           "تحسين الصورة، وتصويت بين عدة محركات ونسخ معالجة.")

# التحقق من Tesseract
if not (shutil.which("tesseract") or
        pytesseract.pytesseract.tesseract_cmd != "tesseract"
        and shutil.which(pytesseract.pytesseract.tesseract_cmd)):
    st.error("Tesseract غير مثبت. على Streamlit Cloud أضف `tesseract-ocr` إلى "
             "ملف packages.txt. على ويندوز ثبّته من موقع UB-Mannheim.")
    st.stop()

try:
    available = pytesseract.get_languages(config="")
except Exception:
    available = ["eng"]

with st.sidebar:
    st.header("⚙️ الإعدادات")
    engines_opts = ["Tesseract"] + (["EasyOCR"] if HAS_EASYOCR else [])
    engines = st.multiselect("محركات القراءة", engines_opts, default=engines_opts)
    if not HAS_EASYOCR:
        st.caption("EasyOCR غير مثبت (اختياري، يرفع الدقة). راجع requirements.txt.")
    lang_opts = [l for l in ("eng", "ara") if l in available] or ["eng"]
    langs = st.multiselect("لغة اللوحة", lang_opts, default=lang_opts[:1],
                           help="eng = إنجليزي/أرقام، ara = عربي")
    st.subheader("التحسين")
    denoise = st.checkbox("إزالة الضوضاء", True)
    lowlight = st.checkbox("تصحيح الإضاءة الضعيفة", True)
    sharpen = st.checkbox("إزالة الضبابية (شحذ)", True)
    deskew_on = st.checkbox("تعديل ميلان النص", True)
    st.subheader("الكشف")
    n_cands = st.slider("عدد مرشحي اللوحة للتجربة", 1, 3, 2)
    whole = st.checkbox("الصورة هي اللوحة نفسها (بدون كشف)", False)
    pattern = st.text_input("نمط اللوحة (اختياري)", "",
                            help="D = رقم، L = حرف. مثال: LLDDDD. "
                                 "يصحح الالتباس مثل O↔0 و B↔8.")

uploaded = st.file_uploader("ارفع صورة سيارة أو لوحة",
                            type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is None:
    st.info("👆 ارفع صورة للبدء.")
    st.stop()
if not engines or not langs:
    st.warning("اختر محرك قراءة ولغة واحدة على الأقل من القائمة الجانبية.")
    st.stop()

opts = {"engines": engines, "langs": langs, "denoise": denoise,
        "lowlight": lowlight, "sharpen": sharpen, "deskew": deskew_on}

image = load_image(uploaded)

with st.spinner("جاري التحليل..."):
    if whole:
        plates = [(1.0, None, image)]
    else:
        found = []
        if yolo_available():
            try:
                found = detect_with_yolo(image, top_n=n_cands)
            except Exception as e:
                st.warning(f"تعذر تشغيل نموذج YOLO: {e}")
        if not found:  # احتياط: الكشف الكلاسيكي
            found = detect_plates(image, top_n=n_cands)
        plates = [(s, b, warp_plate(image, b)) for s, b in found]
        if not plates:
            st.warning("لم يتم كشف لوحة تلقائياً، سيتم تحليل الصورة كاملة. "
                       "جرّب خيار «الصورة هي اللوحة نفسها» بعد قص اللوحة.")
            plates = [(0.0, None, image)]

    results = []
    for score, box, crop in plates:
        r = read_plate(crop, opts)
        text, conf, agree, total = consensus(r["cands"])
        rank = conf * (0.5 + agree / total) if total else 0
        results.append({"box": box, "crop": crop, "res": r, "text": text,
                        "conf": conf, "agree": agree, "total": total,
                        "rank": rank})

best = max(results, key=lambda x: x["rank"])
final_text = apply_pattern(best["text"], pattern) if best["text"] else ""

# عرض النتيجة النهائية
st.divider()
if final_text:
    st.success(f"### رقم اللوحة:  `{final_text}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("متوسط الثقة", f"{best['conf']:.1f}%")
    c2.metric("اتفاق القراءات", f"{best['agree']} من {best['total']}")
    c3.metric("حدة الصورة (Laplacian)", f"{best['res']['blur']:.0f}")
    if best["conf"] < 60 or best["agree"] < max(2, best["total"] // 3):
        st.warning("الثقة منخفضة. جرّب صورة أوضح، أو فعّل EasyOCR، "
                   "أو حدد نمط اللوحة، أو قص اللوحة يدوياً.")
else:
    st.error("لم يتم التعرف على أي نص. جرّب صورة أوضح أو قص اللوحة يدوياً.")

# مراحل المعالجة
tab1, tab2, tab3, tab4 = st.tabs(["الكشف", "اللوحة المحسّنة",
                                  "نسخ المعالجة", "كل القراءات"])
with tab1:
    vis = image.copy()
    for r in results:
        if r["box"] is not None:
            cv2.polylines(vis, [r["box"].astype(np.int32)], True,
                          (0, 255, 0) if r is best else (0, 165, 255), 3)
    st.image(to_rgb(vis), caption="الأخضر = اللوحة المختارة",
             use_container_width=True)
with tab2:
    a, b = st.columns(2)
    a.image(to_rgb(best["crop"]), caption="اللوحة بعد تصحيح المنظور",
            use_container_width=True)
    b.image(best["res"]["enhanced"], caption="بعد التحسين",
            use_container_width=True)
with tab3:
    cols = st.columns(len(best["res"]["variants"]))
    for col, (name, im) in zip(cols, best["res"]["variants"].items()):
        col.image(im, caption=name, use_container_width=True)
with tab4:
    df = pd.DataFrame(best["res"]["cands"]).drop(columns=["score"], errors="ignore")
    if df.empty:
        st.write("لا توجد قراءات.")
    else:
        df = df.sort_values("conf", ascending=False)
        st.dataframe(df, use_container_width=True)

if final_text:
    out = pd.DataFrame([{"plate": final_text,
                         "confidence": round(best["conf"], 1),
                         "agreement": f"{best['agree']}/{best['total']}"}])
    st.download_button("تنزيل النتيجة (CSV)",
                       out.to_csv(index=False).encode("utf-8-sig"),
                       "plate_result.csv", "text/csv")
