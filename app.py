import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import pandas as pd
import database as db

# تحديد مسار محرك Tesseract على نظام Linux الخاص بـ Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# تهيئة قاعدة البيانات
try:
    db.init_db()
except Exception:
    pass

st.set_page_config(
    page_title="نظام ALPR المستقر للتعرف على اللوحات",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 نظام التعرف وإدارة لوحات السيارات (ALPR System)")

# وظيفة تحسين جودة الصورة للـ OCR باستخدام Pillow فقط (بدون OpenCV)
def preprocess_image_pil(image):
    # تحويل الصورة إلى أبيض وأسود (Grayscale)
    gray = image.convert('L')
    # زيادة التباين (Contrast) لتوضيح الأرقام
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.0)
    return enhanced

# التبويبات الرئيسية
tab1, tab2 = st.tabs(["🔍 فحص وتحليل صورة", "📊 سجل اللوحات المكتشفة"])

with tab1:
    st.subheader("رفع صورة السيارة واستخراج رقم اللوحة")
    uploaded_file = st.file_uploader("قم برفع صورة السيارة هنا...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="الصورة المرفوعة", use_container_width=True)

        if st.button("بدء عملية الفحص واستخراج رقم اللوحة", type="primary"):
            with st.spinner("جاري معالجة الصورة واستخراج النص..."):
                # معالجة الصورة أولاً لزيادة دقة القراءة
                processed_img = preprocess_image_pil(image)
                
                # إعدادات Tesseract لقراءة الأرقام والحروف بأعلى دقة
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(processed_img, config=custom_config)
                
                # تنظيف النص المستخرج
                clean_text = "".join([c for c in text if c.isalnum() or c in ['-', ' ']]).strip()

                if len(clean_text) > 1:
                    db.insert_plate(clean_text, 90.0)
                    st.success(f"✅ **تم استخراج رقم اللوحة بنجاح:** `{clean_text}`")
                    with col2:
                        st.image(processed_img, caption="الصورة المعالجة للـ OCR", use_container_width=True)
                else:
                    st.warning("⚠️ لم يتم العثور على أرقام أو نصوص واضحة، حاول رفع صورة أوضح للوحة.")

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
