import streamlit as st
import numpy as np
from PIL import Image
import pytesseract
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

# 2. تحميل نموذج YOLO
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# 3. الواجهة الرئيسية
tab1, tab2 = st.tabs(["🔍 فحص صورة جديدة", "📊 سجل السيارات المكتشفة"])

with tab1:
    uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        
        col1, col2 = st.columns(2)
        col1.image(image, caption="الصورة الأصلية", use_container_width=True)
            
        if st.button("تحليل الصورة واستخراج اللوحة"):
            with st.spinner("جاري الكشف واستخراج النص..."):
                yolo_results = model(img_np)
                boxes = yolo_results[0].boxes
                plate_found = False

                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        plate_crop_pil = image.crop((x1, y1, x2, y2))
                        
                        if plate_crop_pil.size[0] > 0 and plate_crop_pil.size[1] > 0:
                            # استخراج النص باستخدام Tesseract
                            text = pytesseract.image_to_string(plate_crop_pil, config='--psm 6')
                            clean_text = "".join([c for c in text if c.isalnum() or c in ['-', ' ']]).strip()
                            
                            if len(clean_text) > 1:
                                plate_found = True
                                db.insert_plate(clean_text, 95.0)
                                st.success(f"**النص المكتشف:** `{clean_text}`")
                                col2.image(plate_crop_pil, caption="اللوحة المقتطعة", use_container_width=True)
                                break

                if not plate_found:
                    # محاولة قراءة الصورة كاملة في حال عدم العثور عبر YOLO
                    text = pytesseract.image_to_string(image, config='--psm 6')
                    clean_text = "".join([c for c in text if c.isalnum() or c in ['-', ' ']]).strip()
                    if len(clean_text) > 1:
                        db.insert_plate(clean_text, 85.0)
                        st.success(f"**النص المكتشف:** `{clean_text}`")
                        col2.image(yolo_results[0].plot(), caption="تحديد الذكاء الاصطناعي (YOLO)", use_container_width=True)
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
