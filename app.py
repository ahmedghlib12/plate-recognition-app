import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import pandas as pd
import database as db

# 1. تهيئة قاعدة البيانات
db.init_db()

st.set_page_config(page_title="نظام ALPR المتكامل", layout="wide")
st.title("🚗 نظام التعرف وإدارة لوحات السيارات (YOLO + OpenCV + SQLite)")

# 2. تحميل النماذج بالذاكرة مرة واحدة
@st.cache_resource
def load_models():
    # استخدام نموذج YOLO المتاح في المجلد
    yolo_model = YOLO("yolov8n.pt") 
    ocr_reader = easyocr.Reader(['en', 'ar'])
    return yolo_model, ocr_reader

model, reader = load_models()

# 3. دالة معالجة الظروف الصعبة (الإضاءة والضبابية)
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(filtered, -1, kernel)
    return sharpened

# 4. واجهة التطبيق بتبويبين
tab1, tab2 = st.tabs(["🔍 فحص صورة جديدة", "📊 سجل السيارات المكتشفة"])

with tab1:
    uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="الصورة الأصلية", use_container_width=True)
            
        if st.button("تحليل الصورة واستخراج اللوحة"):
            with st.spinner("جاري الكشف بـ YOLO وقراءة النص بـ OCR..."):
                # كشف الأجسام باستخدام YOLO
                yolo_results = model(img_np)
                
                # تطبيق تحسين الصورة المتقدم
                processed_img = preprocess_image(img_np)
                ocr_results = reader.readtext(processed_img)
                
                if ocr_results:
                    st.success("تم التعرف على النص وتحسينه بنجاح!")
                    for bbox, text, prob in ocr_results:
                        if prob > 0.20 and len(text.strip()) > 1:
                            # حفظ السجل في قاعدة البيانات
                            db.insert_plate(text, float(prob * 100))
                            st.write(f"- النص المكتشف: {text} | نسبة الثقة: {prob*100:.1f}%")
                    
                    with col2:
                        st.image(yolo_results[0].plot(), caption="تحديد الذكاء الاصطناعي (YOLO)", use_container_width=True)
                else:
                    st.warning("لم يتم العثور على أرقام أو نصوص واضحة.")

with tab2:
    st.subheader("سجلات السيارات المحفوظة في قاعدة البيانات")
    logs = db.get_all_logs()
    
    if logs:
        df = pd.DataFrame(logs, columns=["المعرف (ID)", "رقم اللوحة", "نسبة الدقة %", "تاريخ ووقت الاكتشاف"])
        st.dataframe(df, use_container_width=True)
        
        # زر إضافي لتنزيل السجلات كملف CSV
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 تحميل السجل كملف CSV",
            data=csv,
            file_name='plate_logs.csv',
            mime='text/csv',
        )
    else:
        st.info("لا توجد سجلات مخزنة حتى الآن.")