import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import base64
import yt_dlp
import os
from ultralytics import YOLO

st.set_page_config(page_title="أرصاد الحشود | Crowd Analytics", page_icon="🕋", layout="wide")

# 1. تحميل محرك الذكاء الاصطناعي الحقيقي (يُحمل مرة واحدة فقط في الذاكرة)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

ai_model = load_model()

# 2. إعدادات المعايرة (ROI & Calibration)
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ غرفة المعايرة</h2>", unsafe_allow_html=True)
st.sidebar.markdown("**تخصيص الكاميرا والسعة:**")
horizon_cutoff = st.sidebar.slider("✂️ اقتطاع الأفق (إخفاء السماء/المآذن) %", 0, 70, 30)
max_capacity = st.sidebar.number_input("👥 السعة القصوى للمنطقة (للمعايرة)", min_value=100, max_value=10000, value=1500, step=100)
conf_thresh = st.sidebar.slider("🎯 دقة الرصد (Confidence)", 0.05, 0.80, 0.15, 0.05)

def get_b64(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

safe_bg = get_b64("~~~.jpg")
alert_bg = get_b64("~~~~.jpg")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} [data-testid="stHeader"] {visibility: hidden;}
    .block-container { padding-top: 1rem !important; }
    
    .stImage > img {
        max-height: 380px; 
        object-fit: contain;
        border-radius: 10px;
        border: 1px solid #333;
    }
    
    .metric-card {
        border-radius: 12px;
        padding: 20px 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        background: linear-gradient(135deg, #1e1e1e, #2a2a2a);
        position: relative;
        overflow: hidden;
        border: 2px solid #444;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-title { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 1.1rem; font-weight: 600; margin-bottom: 5px; color: #ddd; }
    
    .metric-val { 
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 2.5rem; 
        font-weight: 900; 
        text-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5); 
    }
    
    .status-safe { color: #00fa9a; border-color: #00fa9a; }
    .status-alert { color: #ff4b4b; border-color: #ff4b4b; }
    .status-mid { color: #ffd700; border-color: #ffd700; }
</style>
""", unsafe_allow_html=True)

def analyze_ai_frame(frame, horizon_pct, max_cap, conf):
    h, w = frame.shape[:2]
    
    # تحديد منطقة الاهتمام (ROI) وتجاهل السماء تماماً
    cutoff_y = int(h * (horizon_pct / 100.0))
    roi_frame = frame.copy()
    roi_frame[:cutoff_y, :] = 0 # تعمية الذكاء الاصطناعي عن السماء
    
    # تشغيل العقل الاصطناعي (رصد الأشخاص فقط - Class 0)
    results = ai_model.predict(roi_frame, classes=[0], conf=conf, verbose=False)
    
    person_count = 0
    overlay = frame.copy()
    
    # رسم المربعات حول البشر حصراً
    for box in results[0].boxes:
        person_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 150), 2)
        # نقطة السنتر لتتبع الكثافة
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)

    # تظليل منطقة السماء المقتطعة لتوضيحها للمستخدم
    cv2.rectangle(overlay, (0, 0), (w, cutoff_y), (0, 0, 0), -1)
    
    # حساب المقاييس الحقيقية
    occupancy_pct = min(int((person_count / max_cap) * 100), 100)
    
    if occupancy_pct < 40:
        level_text, color_class, hex_color = "منخفض", "status-safe", "#00fa9a"
    elif occupancy_pct < 75:
        level_text, color_class, hex_color = "متوسط", "status-mid", "#ffd700"
    else:
        level_text, color_class, hex_color = "عالي / حرج", "status-alert", "#ff4b4b"
        
    res = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)
    
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), person_count, occupancy_pct, level_text, hex_color

def generate_clean_card(title, val, hex_color):
    return f'''
    <div class="metric-card" style="border-color: {hex_color};">
        <div class="metric-title">{title}</div>
        <div class="metric-val" style="color: {hex_color};">{val}</div>
    </div>
    '''

def update_realtime_ui(count, occ_pct, level_text, hex_color, placeholders):
    placeholders[0].markdown(generate_clean_card("العدد التقديري (أشخاص)", count, hex_color), unsafe_allow_html=True)
    placeholders[1].markdown(generate_clean_card("نسبة الإشغال", f"{occ_pct}%", hex_color), unsafe_allow_html=True)
    placeholders[2].markdown(generate_clean_card("مستوى الكثافة", level_text, hex_color), unsafe_allow_html=True)

st.title("🕋 المنصة الذكية لأرصاد الحشود")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["صورة ثابتة", "فيديو مسجل", "بث مباشر (يوتيوب)"])

with tab1:
    uploaded_file = st.file_uploader("📂 ارفع صورة...", type=["jpg", "jpeg", "png"], key="img_upload")
    if uploaded_file:
        img_array = cv2.cvtColor(np.array(Image.open(uploaded_file).convert('RGB')), cv2.COLOR_RGB2BGR)
        
        c_img1, c_img2 = st.columns([1, 1])
        with c_img1:
            st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True, caption="الرؤية الأصلية")
        with c_img2:
            img_ph = st.empty()
            
        st.write("")
        col1, col2, col3 = st.columns(3)
        dash_ph = [col1.empty(), col2.empty(), col3.empty()]
        
        if st.button("🚀 بدء التحليل الدقيق", key="btn_img"):
            out_rgb, p_count, occ_pct, lvl_txt, h_color = analyze_ai_frame(img_array, horizon_cutoff, max_capacity, conf_thresh)
            img_ph.image(out_rgb, use_container_width=True, caption="رصد الذكاء الاصطناعي (YOLOv8)")
            update_realtime_ui(p_count, occ_pct, lvl_txt, h_color, dash_ph)

with tab2:
    uploaded_video = st.file_uploader("📂 ارفع مقطع فيديو (MP4)...", type=["mp4"], key="vid_upload")
    if uploaded_video:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_video.read())
                temp_path = tfile.name
                
            img_ph = st.empty()
            col1, col2, col3 = st.columns(3)
            dash_ph = [col1.empty(), col2.empty(), col3.empty()]
            
            if st.button("🚀 تشغيل الأرصاد", key="btn_vid"):
                cap = cv2.VideoCapture(temp_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                process_interval = max(1, fps // 2) # معالجة إطارين في الثانية لتخفيف الضغط
                
                count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    count += 1
                    if count % process_interval != 0: continue
                        
                    out_rgb, p_count, occ_pct, lvl_txt, h_color = analyze_ai_frame(frame, horizon_cutoff, max_capacity, conf_thresh)
                    img_ph.image(out_rgb, use_container_width=True)
                    update_realtime_ui(p_count, occ_pct, lvl_txt, h_color, dash_ph)
                cap.release()
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

with tab3:
    youtube_url = st.text_input("🔗 أدخل رابط يوتيوب:", key="yt_url")
    if youtube_url:
        img_ph = st.empty()
        col1, col2, col3 = st.columns(3)
        dash_ph = [col1.empty(), col2.empty(), col3.empty()]
        
        if st.button("🚀 بدء الاستشعار الحي", key="btn_live"):
            try:
                ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    stream_url = info['url']
                
                cap = cv2.VideoCapture(stream_url)
                fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
                process_interval = max(1, fps) # معالجة إطار واحد كل ثانية للبث المباشر
                
                count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    count += 1
                    if count % process_interval != 0: continue
                        
                    out_rgb, p_count, occ_pct, lvl_txt, h_color = analyze_ai_frame(frame, horizon_cutoff, max_capacity, conf_thresh)
                    img_ph.image(out_rgb, use_container_width=True)
                    update_realtime_ui(p_count, occ_pct, lvl_txt, h_color, dash_ph)
                cap.release()
            except Exception as e:
                st.error(f"حدث خطأ أثناء الاتصال بالبث المباشر: {e}")
