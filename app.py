import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import base64
from collections import deque
from ultralytics import YOLO
import yt_dlp

st.set_page_config(page_title="أرصاد الحشود | Crowd Analytics", page_icon="🕋", layout="wide")

if 'density_history' not in st.session_state:
    st.session_state.density_history = deque(maxlen=5)

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

safe_bg = get_base64_of_bin_file("~.jpg")
alert_bg = get_base64_of_bin_file("~~.jpg")
men_bg = get_base64_of_bin_file("~~~.jpg")
women_bg = get_base64_of_bin_file("~~~~.jpg")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} [data-testid="stHeader"] {visibility: hidden;}
    .block-container { padding-top: 2rem !important; }
    
    .metric-card {
        border-radius: 15px;
        padding: 30px 20px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.5);
        background-size: cover;
        background-position: center;
        position: relative;
        overflow: hidden;
        border: 2px solid #333;
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.65);
        z-index: 1;
    }
    .metric-content {
        position: relative;
        z-index: 2;
    }
    .metric-title { font-size: 1.2rem; font-weight: 600; margin-bottom: 15px; color: #e0e0e0; }
    .metric-val { font-size: 3.5rem; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.8); }
    .trend-arrow { font-size: 1.5rem; margin-left: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return YOLO("yolov8s.pt")

model = load_engine()

def analyze_crowd_logic(frame_bgr, conf=0.15, img_size=640):
    results = model.predict(frame_bgr, classes=[0], conf=conf, imgsz=img_size, verbose=False)
    
    men_count, women_count = 0, 0
    total_people = len(results[0].boxes)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w = y2 - y1, x2 - x1
        
        if h < 10 or w < 10: continue
            
        head_y_end = y1 + int(h * 0.3)
        head_crop = frame_bgr[y1:head_y_end, x1:x2]
        torso_crop = frame_bgr[head_y_end:y2, x1:x2]
        
        if head_crop.size == 0 or torso_crop.size == 0: continue
            
        hsv_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        hsv_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 60])
        
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        
        head_dark = cv2.countNonZero(cv2.inRange(hsv_head, lower_dark, upper_dark))
        head_skin = cv2.countNonZero(cv2.inRange(hsv_head, lower_skin, upper_skin))
        head_white = cv2.countNonZero(cv2.inRange(hsv_head, lower_white, upper_white))
        
        torso_white = cv2.countNonZero(cv2.inRange(hsv_torso, lower_white, upper_white))
        torso_dark = cv2.countNonZero(cv2.inRange(hsv_torso, lower_dark, upper_dark))
        
        head_pixels = head_crop.shape[0] * head_crop.shape[1]
        torso_pixels = torso_crop.shape[0] * torso_crop.shape[1]
        
        if (head_dark > head_pixels*0.1 or head_skin > head_pixels*0.1 or head_white > head_pixels*0.1) and (torso_white > torso_pixels*0.2):
            men_count += 1
            color = (255, 255, 255)
            
        elif (head_dark > head_pixels*0.2) and (torso_dark > torso_pixels*0.2):
            women_count += 1
            color = (255, 0, 255)
            
        else:
            color = (0, 255, 0)
            
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        
    MAX_CAPACITY = 800 
    raw_density = min(int((total_people / MAX_CAPACITY) * 100), 100)
    
    classified_total = men_count + women_count
    men_ratio = int((men_count / classified_total) * 100) if classified_total > 0 else 0
    women_ratio = 100 - men_ratio if classified_total > 0 else 0
        
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), raw_density, men_ratio, women_ratio

def render_dashboard(density, m_r, w_r, placeholders, is_dynamic=True):
    trend_arrow = "➖"
    
    if is_dynamic:
        st.session_state.density_history.append(density)
        avg_density = int(sum(st.session_state.density_history) / len(st.session_state.density_history))
        
        if len(st.session_state.density_history) >= 2:
            prev_avg = st.session_state.density_history[-2]
            if avg_density > prev_avg + 1:
                trend_arrow = "⬆️"
            elif avg_density < prev_avg - 1:
                trend_arrow = "⬇️"
        display_density = avg_density
    else:
        display_density = density
        trend_arrow = ""

    current_density_bg = alert_bg if display_density > 70 else safe_bg
    val_color = "#ff4b4b" if display_density > 70 else "#00fa9a"
    
    html_density = f'''
        <div class="metric-card" style="background-image: url('data:image/jpeg;base64,{current_density_bg}'); border-color: {val_color};">
            <div class="metric-content">
                <div class="metric-title">مؤشر أرصاد الكثافة اللحظي</div>
                <div class="metric-val" style="color: {val_color};">{display_density}% <span class="trend-arrow">{trend_arrow}</span></div>
            </div>
        </div>
    '''
    
    html_men = f'''
        <div class="metric-card" style="background-image: url('data:image/jpeg;base64,{men_bg}');">
            <div class="metric-content">
                <div class="metric-title">نسبة الرجال (إحرام / وطني)</div>
                <div class="metric-val" style="color: #ffffff;">{m_r}%</div>
            </div>
        </div>
    '''
    
    html_women = f'''
        <div class="metric-card" style="background-image: url('data:image/jpeg;base64,{women_bg}');">
            <div class="metric-content">
                <div class="metric-title">نسبة النساء (عباءات)</div>
                <div class="metric-val" style="color: #4da6ff;">{w_r}%</div>
            </div>
        </div>
    '''
    
    placeholders[0].markdown(html_density, unsafe_allow_html=True)
    placeholders[1].markdown(html_men, unsafe_allow_html=True)
    placeholders[2].markdown(html_women, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3340/3340156.png", width=60)
    st.title("⚙️ الإعدادات")
    source_type = st.radio("مصدر البيانات:", ["صورة ثابتة", "فيديو مسجل", "بث مباشر (يوتيوب)"])

st.title("🕋 المنصة الذكية لأرصاد الحشود")
st.markdown("---")

if source_type == "صورة ثابتة":
    uploaded_file = st.file_uploader("📂 ارفع صورة...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_array = cv2.cvtColor(np.array(Image.open(uploaded_file).convert('RGB')), cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True, caption="اللقطة الأصلية")
        with col2:
            img_placeholder = st.empty()
            
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        dash_placeholders = [c1.empty(), c2.empty(), c3.empty()]
        
        if st.button("🚀 بدء تحليل الزحام"):
            with st.spinner("جاري التحليل..."):
                out_rgb, dens, m_r, w_r = analyze_crowd_logic(img_array, img_size=1280)
                img_placeholder.image(out_rgb, use_container_width=True, caption="خريطة الرصد")
                render_dashboard(dens, m_r, w_r, dash_placeholders, is_dynamic=False)

elif source_type == "فيديو مسجل":
    uploaded_video = st.file_uploader("📂 ارفع مقطع فيديو (MP4)...", type=["mp4"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        img_placeholder = st.empty()
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        dash_placeholders = [c1.empty(), c2.empty(), c3.empty()]
        
        if st.button("🚀 تشغيل الأرصاد اللحظية"):
            cap = cv2.VideoCapture(tfile.name)
            count = 0
            st.session_state.density_history.clear()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                count += 1
                if count % 5 != 0: continue
                    
                out_rgb, dens, m_r, w_r = analyze_crowd_logic(frame)
                img_placeholder.image(out_rgb, use_container_width=True)
                render_dashboard(dens, m_r, w_r, dash_placeholders, is_dynamic=True)

elif source_type == "بث مباشر (يوتيوب)":
    youtube_url = st.text_input("🔗 أدخل رابط يوتيوب:")
    if youtube_url:
        img_placeholder = st.empty()
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        dash_placeholders = [c1.empty(), c2.empty(), c3.empty()]
        
        if st.button("🚀 بدء الاستشعار الحي"):
            try:
                ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    stream_url = info['url']
                
                cap = cv2.VideoCapture(stream_url)
                count = 0
                st.session_state.density_history.clear()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    count += 1
                    if count % 30 != 0: continue
                        
                    out_rgb, dens, m_r, w_r = analyze_crowd_logic(frame)
                    img_placeholder.image(out_rgb, use_container_width=True)
                    render_dashboard(dens, m_r, w_r, dash_placeholders, is_dynamic=True)
            except Exception as e:
                st.error("❌ حدث خطأ في سحب البث.")
