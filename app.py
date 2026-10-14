import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import pandas as pd
import database as db

# مسار محرك Tesseract المعتمد في بيئة Linux على Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# تهيئة قاعدة البيانات
try:
    db.init_db()
except Exception:
    pass

st.set_page_config(page_title="نظام ALPR المستقر", page_icon="🚗", layout="wide")
st.title("🚗 نظام التعرف وإدارة لوحات السيارات (ALPR System)")

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
                try:
                    # تحسين الصورة باستخدام Pillow المدمجة في بايثون
                    gray_img = ImageOps.grayscale(image)
                    enhanced_img = ImageEnhance.Contrast(gray_img).enhance(2.0)

                    # استخراج النص عبر Tesseract مباشرة
                    text = pytesseract.image_to_string(enhanced_img, config='--oem 3 --psm 6')
                    clean_text = "".join([c for c in text if c.isalnum() or c in ['-', ' ']]).strip()

                    if len(clean_text) >= 2:
                        db.insert_plate(clean_text, 90.0)
                        st.success(f"✅ **تم استخراج رقم اللوحة بنجاح:** `{clean_text}`")
                        with col2:
                            st.image(enhanced_img, caption="الصورة المعالجة للـ OCR", use_container_width=True)
                    else:
                        # محاولة قراءة الصورة الأصلية بدون تحسين في حال عدم التمكن
                        raw_text = pytesseract.image_to_string(image, config='--oem 3 --psm 3')
                        raw_clean = "".join([c for c in raw_text if c.isalnum() or c in ['-', ' ']]).strip()
                        
                        if len(raw_clean) >= 2:
                            db.insert_plate(raw_clean, 85.0)
                            st.success(f"✅ **تم استخراج رقم اللوحة بنجاح:** `{raw_clean}`")
                        else:
                            st.warning("⚠️ لم يتم العثور على نص واضح، يرجى تجربة رفع صورة أقرب للوحة السيارة.")
                except Exception as e:
                    st.error(f"حدث خطأ أثناء المعالجة: {str(e)}")

with tab2:
    st.subheader("سجلات قاعدة البيانات (Database Logs)")
    try:
        logs = db.get_all_logs()
        if logs:
            df = pd.DataFrame(logs, columns=["المعرف", "رقم اللوحة", "نسبة الدقة %", "تاريخ ووقت التسجيل"])
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 تحميل السجل CSV", data=csv, file_name='plates_log.csv', mime='text/csv')
        else:
            st.info("لا توجد سجلات محفوظة حالياً.")
    except Exception:
        st.error("تعذر الاتصال بقاعدة البيانات.")
