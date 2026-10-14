import streamlit as st
import numpy as np
from PIL import Image
import pytesseract
import pytesseract

# تحديد مسار tesseract على سيرفرات Linux السحابية
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
from ultralytics import YOLO
import pandas as pd
import database as db

# تهيئة قاعدة البيانات
try:
    db.init_db()
except Exception as e:
    pass

st.set_page_config(
    page_title="نظام ALPR الذكي للتعرف على اللوحات",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 نظام التعرف وإدارة لوحات السيارات (ALPR System)")

# تحميل نموذج YOLOv8 مع التخزين المؤقت
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

# إنشاء التبويبات
tab1, tab2 = st.tabs(["🔍 فحص وتحليل صورة", "📊 سجل اللوحات المكتشفة"])

with tab1:
    st.subheader("تحليل الصور واستخراج اللوحات")
    uploaded_file = st.file_uploader("قم برفع صورة السيارة هنا...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="الصورة المرفوعة", use_container_width=True)

        if st.button("بدء عملية الفحص والاستخراج", type="primary"):
            with st.spinner("جاري تحليل الصورة بواسطة YOLOv8 واستخراج النص عبر Tesseract..."):
                results = model(img_np)
                boxes = results[0].boxes
                plate_detected = False

                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # الاقتطاع بواسطة PIL مباشرة
                        cropped_pil = image.crop((x1, y1, x2, y2))

                        if cropped_pil.size[0] > 0 and cropped_pil.size[1] > 0:
                            # القراءة عبر Tesseract OCR
                            text = pytesseract.image_to_string(cropped_pil, config='--psm 6')
                            clean_text = "".join([c for c in text if c.isalnum() or c in ['-', ' ']]).strip()

                            if len(clean_text) > 1:
                                plate_detected = True
                                db.insert_plate(clean_text, 95.0)
                                st.success(f"✅ **تم اكتشاف اللوحة بنجاح:** `{clean_text}`")
                                with col2:
                                    st.image(cropped_pil, caption="اللوحة المقتطعة", use_container_width=True)
                                break

                if not plate_detected:
                    # محاولة قراءة الصورة كاملاً إذا لم يقطع YOLO لوحة دقيقة
                    text = pytesseract.image_to_string(image, config='--psm 6')
                    clean_text = "".join([c for c in text if c.isalnum() or c in ['-', ' ']]).strip()

                    if len(clean_text) > 1:
                        db.insert_plate(clean_text, 80.0)
                        st.success(f"✅ **النص المستخرج من الصورة:** `{clean_text}`")
                    else:
                        st.warning("⚠️ لم يتم العثور على أرقام أو نصوص واضحة في الصورة.")

with tab2:
    st.subheader("سجلات قاعدة البيانات (Database Logs)")
    try:
        logs = db.get_all_logs()
        if logs:
            df = pd.DataFrame(logs, columns=["المعرف", "رقم اللوحة", "نسبة الدقة %", "تاريخ ووقت التسجيل"])
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 تحميل السجل كملف CSV",
                data=csv,
                file_name='plates_log.csv',
                mime='text/csv'
            )
        else:
            st.info("لا توجد سجلات محفوظة حالياً.")
    except Exception:
        st.error("تعذر الاتصال بقاعدة البيانات.")
