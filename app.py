import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import yt_dlp
import time

st.set_page_config(page_title="أرصاد الحشود | Crowd Analytics", page_icon="🕋", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    
    .block-container { padding-top: 2rem !important; }
    
    .metric-box {
        background-color: #1e1e2f;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-left: 5px solid #d4af37;
        margin-bottom: 15px;
    }
    .metric-title { color: #a0a0b0; font-size: 1.1rem; margin-bottom: 10px; }
    .metric-val { font-size: 2.2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 1. تحميل النموذج المركزي
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# 2. إعدادات الواجهة
col_settings, col_empty = st.columns([1, 10])
with col_settings:
    with st.popover("⚙️ إعدادات النظام"):
        source_type = st.radio("مصدر البيانات:", ["صورة ثابتة", "فيديو مسجل", "بث مباشر (يوتيوب)"], label_visibility="collapsed")
        conf_thresh = st.slider("دقة الرصد (Confidence):", 0.05, 1.0, 0.15, 0.05)
        # خيار لدقة المعالجة، نجعله 640 للفيديو لتجنب التعليق، و1280 للصور
        ai_res = st.selectbox("دقة فحص الذكاء الاصطناعي:", [640, 1280], index=0)

st.title("🕋 المنصة الذكية لأرصاد الحشود")
st.markdown("### نظام الاستشعار الموحد (Unified AI Pipeline) للصور والفيديو والبث اللحظي")
st.markdown("---")

# 3. العقل المركزي (الدالة الموحدة التي تحلل أي لقطة تمرر لها)
def process_frame(frame_bgr, conf, img_size):
    # الفحص عبر YOLO
    results = model.predict(frame_bgr, classes=[0], conf=conf, imgsz=img_size, verbose=False)
    
    men_count = 0
    women_count = 0
    total_people = len(results[0].boxes)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_crop = frame_bgr[y1:y2, x1:x2]
        
        if person_crop.size == 0: continue
            
        hsv_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        
        # تحليل الألوان
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv_crop, lower_white, upper_white)
        
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 70])
        mask_black = cv2.inRange(hsv_crop, lower_black, upper_black)
        
        white_px = cv2.countNonZero(mask_white)
        black_px = cv2.countNonZero(mask_black)
        
        if white_px > black_px:
            men_count += 1
            color = (255, 255, 255) 
        else:
            women_count += 1
            color = (255, 0, 0) 
            
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    MAX_CAPACITY = 500
    density = min(int((total_people / MAX_CAPACITY) * 100), 100)
    
    men_ratio = int((men_count / total_people) * 100) if total_people > 0 else 0
    women_ratio = 100 - men_ratio if total_people > 0 else 0
        
    return frame_rgb, density, total_people, men_ratio, women_ratio

# دالة لتحديث لوحة الأرصاد في الواجهة
def update_dashboard(density, total_ppl, men_r, women_r, placeholders):
    color_dens = "#ff4b4b" if density > 70 else "#00fa9a"
    placeholders[0].markdown(f'<div class="metric-box"><div class="metric-title">مؤشر الكثافة</div><div class="metric-val" style="color:{color_dens}">{density}%</div></div>', unsafe_allow_html=True)
    placeholders[1].markdown(f'<div class="metric-box"><div class="metric-title">الأشخاص المرصودين</div><div class="metric-val" style="color:#fff">{total_ppl}</div></div>', unsafe_allow_html=True)
    placeholders[2].markdown(f'<div class="metric-box"><div class="metric-title">رجال (إحرام)</div><div class="metric-val" style="color:#fff">{men_r}%</div></div>', unsafe_allow_html=True)
    placeholders[3].markdown(f'<div class="metric-box"><div class="metric-title">نساء (عباءات)</div><div class="metric-val" style="color:#4da6ff">{women_r}%</div></div>', unsafe_allow_html=True)

# 4. توجيه المدخلات إلى العقل المركزي
if source_type == "صورة ثابتة":
    uploaded_file = st.file_uploader("📂 ارفع صورة...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_array_rgb = np.array(image)
        img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
        
        st.markdown("#### 🎯 لوحة الأرصاد والرصد الآلي")
        img_placeholder = st.empty()
        img_placeholder.image(img_array_rgb, use_container_width=True)
        
        c1, c2, c3, c4 = st.columns(4)
        dash_placeholders = [c1.empty(), c2.empty(), c3.empty(), c4.empty()]
        
        if st.button("🚀 بدء تحليل الزحام"):
            with st.spinner("جاري التحليل..."):
                out_rgb, dens, ppl, m_r, w_r = process_frame(img_array_bgr, conf_thresh, ai_res)
                img_placeholder.image(out_rgb, use_container_width=True)
                update_dashboard(dens, ppl, m_r, w_r, dash_placeholders)

elif source_type == "فيديو مسجل":
    uploaded_video = st.file_uploader("📂 ارفع مقطع فيديو (MP4)...", type=["mp4", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        st.markdown("#### 🎯 الاستشعار الحركي اللحظي")
        img_placeholder = st.empty()
        c1, c2, c3, c4 = st.columns(4)
        dash_placeholders = [c1.empty(), c2.empty(), c3.empty(), c4.empty()]
        
        stop_btn = st.button("⏹️ إيقاف التحليل")
        
        if not stop_btn:
            cap = cv2.VideoCapture(tfile.name)
            frame_skip = 10 # تخطي الإطارات لتسريع العرض
            count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                count += 1
                if count % frame_skip != 0: continue # فحص لقطة وتخطي الباقي لضمان سرعة السيرفر
                    
                out_rgb, dens, ppl, m_r, w_r = process_frame(frame, conf_thresh, ai_res)
                img_placeholder.image(out_rgb, use_container_width=True)
                update_dashboard(dens, ppl, m_r, w_r, dash_placeholders)
                
            cap.release()

elif source_type == "بث مباشر (يوتيوب)":
    youtube_url = st.text_input("🔗 أدخل رابط بث يوتيوب مباشر:")
    if youtube_url:
        st.markdown("#### 🔴 الاستشعار من البث المباشر")
        img_placeholder = st.empty()
        c1, c2, c3, c4 = st.columns(4)
        dash_placeholders = [c1.empty(), c2.empty(), c3.empty(), c4.empty()]
        
        stop_btn = st.button("⏹️ إيقاف البث")
        
        if not stop_btn:
            try:
                ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    stream_url = info['url']
                
                cap = cv2.VideoCapture(stream_url)
                frame_skip = 30 # فحص لقطة واحدة كل ثانية من البث
                count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    count += 1
                    if count % frame_skip != 0: continue
                        
                    out_rgb, dens, ppl, m_r, w_r = process_frame(frame, conf_thresh, ai_res)
                    img_placeholder.image(out_rgb, use_container_width=True)
                    update_dashboard(dens, ppl, m_r, w_r, dash_placeholders)
                    
                cap.release()
            except Exception as e:
                st.error(f"❌ حدث خطأ في سحب البث. قد يكون الرابط محمياً. التفاصيل: {e}")
