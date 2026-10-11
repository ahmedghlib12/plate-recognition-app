import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import pandas as pd
import database as db

# 1. تهيئة قاعدة البيانات
try:
    db.init_db()
except Exception:
    pass

st.set_page_config(page_title="نظام ALPR المتكامل", layout="wide")
st.title("🚗 نظام التعرف وإدارة لوحات السيارات")

# 2. تحميل النماذج
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt") 
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
    except Exception:
        ocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=False)
    return yolo_model, ocr_reader

model, reader = load_models()

# 3. الواجهة الرئيسية
tab1, tab2 = st.tabs(["🔍 فحص صورة جديدة", "📊 سجل السيارات المكتشفة"])

with tab1:
    uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="الصورة الأصلية", use_container_width=True)
            
        if st.button("تحليل الصورة واستخراج اللوحة"):
            with st.spinner("جاري الكشف واقتصاص اللوحة وقراءة النص..."):
                yolo_results = model(img_np)
                boxes = yolo_results[0].boxes
                plate_found = False

                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        plate_crop = img_np[y1:y2, x1:x2]
                        
                        if plate_crop.size > 0:
                            plate_resized = cv2.resize(plate_crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                            ocr_results = reader.readtext(plate_resized)
                            
                            if ocr_results:
                                plate_found = True
                                for bbox, text, prob in ocr_results:
                                    clean_text = "".join([c for c in text if c.isalnum() or c in ['-', ' ']]).strip()
                                    if len(clean_text) > 1:
                                        db.insert_plate(clean_text, float(prob * 100))
                                        st.success(f"النص المكتشف: {clean_text} | نسبة الثقة: {prob*100:.1f}%")
                                
                                with col2:
                                    st.image(plate_crop, caption="اللوحة المقتطعة", use_container_width=True)
                
                if not plate_found:
                    ocr_results = reader.readtext(img_np)
                    if ocr_results:
                        for bbox, text, prob in ocr_results:
                            clean_text = "".join([c for c in text if c.isalnum() or c in ['-', ' ']]).strip()
                            if prob > 0.15 and len(clean_text) > 1:
                                db.insert_plate(clean_text, float(prob * 100))
                                st.success(f"النص المكتشف: {clean_text} | نسبة الثقة: {prob*100:.1f}%")
                        with col2:
                            st.image(yolo_results[0].plot(), caption="تحديد الذكاء الاصطناعي (YOLO)", use_container_width=True)
                    else:
                        st.warning("لم يتم العثور على أرقام أو نصوص واضحة.")

with tab2:
    st.subheader("سجلات السيارات المحفوظة في قاعدة البيانات")
    try:
        logs = db.get_all_logs()
        if logs:
            df = pd.DataFrame(logs, columns=["المعرف (ID)", "رقم اللوحة", "نسبة الدقة %", "تاريخ ووقت الاكتشاف"])
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 تحميل السجل كملف CSV",
                data=csv,
                file_name='plate_logs.csv',
                mime='text/csv'
            )
        else:
            st.info("لا توجد سجلات مخزنة حتى الآن.")
    except Exception:
        st.error("تعذر جلب السجلات من قاعدة البيانات.")
