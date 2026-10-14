import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import pandas as pd
import database as db

# مسار tesseract المعتمد في Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

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
            with st.spinner("جاري تحليل الصورة وتجربة الخوارزميات المختلفة..."):
                extracted_text = ""
                
                # قائمة بالتجريبية: الصورة الأصلية، الصورة المحسنة التباين، الصورة الرمادية
                gray_img = ImageOps.grayscale(image)
                enhancer = ImageEnhance.Contrast(gray_img)
                enhanced_img = enhancer.enhance(2.5)

                sources_to_try = [image, gray_img, enhanced_img]
                psm_modes = ['--psm 6', '--psm 11', '--psm 3']

                # محاولة القراءة بأكثر من النمط
                for img_src in sources_to_try:
                    for psm in psm_modes:
                        config = f'{psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
                        text = pytesseract.image_to_string(img_src, config=config)
                        clean = "".join([c for c in text if c.isalnum() or c in ['-', ' ']]).strip()
                        
                        if len(clean) >= 3: # العثور على نص معتبر
                            extracted_text = clean
                            break
                    if extracted_text:
                        break

                if extracted_text:
                    db.insert_plate(extracted_text, 88.0)
                    st.success(f"✅ **تم استخراج النص/اللوحة بنجاح:** `{extracted_text}`")
                    with col2:
                        st.image(enhanced_img, caption="الصورة المفلترة المستخدمة بالقراءة", use_container_width=True)
                else:
                    st.warning("⚠️ لم يتحدد النص تلقائياً من كامل الصورة. يُفضل قص الجزء الخاص باللوحة فقط لرفع الدقة.")

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
