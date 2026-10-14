from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.detector import find_plate
from src.ocr import available_languages, read_plate
from src.parser import parse_plate
from src.preprocessing import resize_keep_ratio

st.set_page_config(page_title="كشف لوحات السيارات", page_icon="🚗", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("<h2 style='text-align:center'>🚗 كشف واستخراج رقم لوحة السيارة</h2>",
            unsafe_allow_html=True)
st.caption("يعمل مع الصور غير الواضحة أو المطموسة أو التي عليها طين، بمعالجة الصور والذكاء الاصطناعي.")

with st.sidebar:
    st.header("الإعدادات")
    show_debug = st.checkbox("عرض تفاصيل المعالجة", value=False)
    langs = available_languages()
    st.write("لغات OCR المتوفرة:", ", ".join(langs))
    if "ara" not in langs:
        st.warning("حزمة العربية غير مثبتة. تأكد من وجود tesseract-ocr-ara في packages.txt")

source = st.radio("مصدر الصورة", ["رفع صورة", "الكاميرا"], horizontal=True)
file = (st.file_uploader("ارفع صورة السيارة", type=["jpg", "jpeg", "png"])
        if source == "رفع صورة" else st.camera_input("التقط صورة"))

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)

    if st.button("بدء عملية الفحص واستخراج رقم اللوحة", type="primary"):
        try:
            with st.spinner("جاري المعالجة..."):
                img_bgr = resize_keep_ratio(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
                plate, annotated, box = find_plate(img_bgr)
                if plate is None:
                    st.warning("لم يتم العثور على لوحة، سنقرأ الصورة كاملة.")
                    plate = img_bgr
                else:
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             caption="موقع اللوحة المكتشف", use_container_width=True)
                best, attempts = read_plate(plate)

            st.image(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), caption="اللوحة المقصوصة")

            if best:
                parsed = parse_plate(best["text"])
                final = parsed["formatted"] if parsed else best["text"]
                st.success(f"رقم اللوحة: {final}")
                c1, c2 = st.columns(2)
                c1.metric("نسبة الثقة", f"{best['conf']:.0f}%")
                c2.metric("طريقة المعالجة", best["method"].split(" (")[0])
                if parsed:
                    st.write(f"رمز المحافظة: **{parsed['code']}** | الحرف: **{parsed['letter']}** | الرقم: **{parsed['number']}**")
                if best["conf"] < 50:
                    st.info("الثقة منخفضة. جرّب صورة أوضح أو بزاوية أقرب للمواجهة.")
                st.image(best["image"], caption="اللوحة بعد التحسين", clamp=True)

                st.session_state.history.append({
                    "الوقت": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "الرقم": final,
                    "الثقة %": round(best["conf"]),
                })
            else:
                st.error("تعذّر استخراج الرقم. جرّب صورة أوضح.")

            if show_debug:
                st.subheader("كل المحاولات")
                for a in attempts:
                    st.write(f"{a['method']} → `{a['text']}` ({a['conf']:.0f}%)")
                    st.image(a["image"], clamp=True)
        except Exception as e:
            st.error(f"حدث خطأ أثناء المعالجة: {e}")

if st.session_state.history:
    st.divider()
    st.subheader("سجل النتائج")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
    st.download_button("تنزيل السجل CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       "plates.csv", "text/csv")
