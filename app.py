import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import base64
import yt_dlp

st.set_page_config(page_title="أرصاد الحشود | Crowd Analytics", page_icon="🕋", layout="wide")

def get_b64(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

safe_bg = get_b64("~~~.jpg")
alert_bg = get_b64("~~~~.jpg")
men_bg = get_b64("~~.jpg")
women_bg = get_b64("~.jpg")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} [data-testid="stHeader"] {visibility: hidden;}
    .block-container { padding-top: 1rem !important; }
    
    /* 1. تصغير الصور بشكل احترافي */
    .stImage > img {
        max-height: 260px; /* تم التصغير من 400 إلى 260 */
        object-fit: contain;
        border-radius: 10px;
        border: 1px solid #333;
    }
    
    .metric-card {
        border-radius: 12px;
        padding: 25px 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        background-size: cover;
        background-position: center;
        position: relative;
        overflow: hidden;
        border: 2px solid #222;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.75); /* تعميق الظل قليلاً لإبراز الأرقام */
        z-index: 1;
    }
    .metric-content {
        position: relative;
        z-index: 2;
    }
    
    /* تغيير الخط الأساسي بخط عصري */
    .metric-title { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 1.1rem; font-weight: 600; margin-bottom: 5px; color: #ddd; }
    
    /* 2. تأثير الشفافية الاحترافي للأرقام (Glassmorphism & Glow) */
    .metric-val { 
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 3rem; 
        font-weight: 900; 
        opacity: 0.85; /* الشفافية المطلوبة */
        text-shadow: 0px 5px 15px rgba(0, 0, 0, 0.9), 0 0 12px currentColor; /* توهج ذكي بنفس لون النص */
    }
    
    /* 3. الفاصل الاحترافي (خط متدرج ومضيء) */
    .divider-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 160px;
    }
    .divider-line {
        width: 3px;
        height: 70%;
        background: linear-gradient(to bottom, transparent, #d4af37, transparent); /* تدرج ذهبي */
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.8);
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

def analyze_pixels(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 40, 255])
    
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 55])
    
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    w_px = cv2.countNonZero(mask_white)
    d_px = cv2.countNonZero(mask_dark)
    
    total_px = frame.shape[0] * frame.shape[1]
    
    roi_limit = total_px * 0.35
    total_crowd = w_px + d_px
    
    density = min(int((total_crowd / roi_limit) * 100), 100)
    empty = 100 - density
    
   def update_ui(density, empty, m_r, w_r, placeholders):
    card_red = generate_card(alert_bg, "نسبة الازدحام", density, "#ff4b4b")
    card_green = generate_card(safe_bg, "نسبة الفضاوة", empty, "#00fa9a")
    card_men = generate_card(men_bg, "نسبة الرجال", m_r, "#ffffff")
    card_women = generate_card(women_bg, "نسبة النساء", w_r, "#c0c0c0") # اللون الرمادي الفضي
    
    if density >= empty:
        box1, box2 = card_red, card_green
    else:
        box1, box2 = card_green, card_red

    placeholders[0].markdown(box1, unsafe_allow_html=True)
    placeholders[1].markdown(box2, unsafe_allow_html=True)
    
    placeholders[2].markdown('<div class="divider-container"><div class="divider-line"></div></div>', unsafe_allow_html=True)
    
    placeholders[3].markdown(card_men, unsafe_allow_html=True)
    placeholders[4].markdown(card_women, unsafe_allow_html=True)
    return f'''
    <div class="metric-card" style="background-image: url('data:image/jpeg;base64,{bg_img}'); border-color: {color};">
        <div class="metric-content">
            <div class="metric-title">{title}</div>
            <div class="metric-val" style="color: {color};">{val}%</div>
        </div>
    </div>
    '''

def update_ui(density, empty, m_r, w_r, placeholders):
    card_red = generate_card(alert_bg, "نسبة الازدحام", density, "#ff4b4b")
    card_green = generate_card(safe_bg, "نسبة الفضاوة", empty, "#00fa9a")
    card_men = generate_card(men_bg, "نسبة الرجال", m_r, "#ffffff")
    card_women = generate_card(women_bg, "نسبة النساء", w_r, "#4da6ff")
    
    if density >= empty:
        box1, box2 = card_red, card_green
    else:
        box1, box2 = card_green, card_red

    placeholders[0].markdown(box1, unsafe_allow_html=True)
    placeholders[1].markdown(box2, unsafe_allow_html=True)
    placeholders[2].markdown('<div class="divider"></div>', unsafe_allow_html=True)
    placeholders[3].markdown(card_men, unsafe_allow_html=True)
    placeholders[4].markdown(card_women, unsafe_allow_html=True)

st.title("🕋 المنصة الذكية لأرصاد الحشود")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["صورة ثابتة", "فيديو مسجل", "بث مباشر (يوتيوب)"])

with tab1:
    uploaded_file = st.file_uploader("📂 ارفع صورة...", type=["jpg", "jpeg", "png"], key="img_upload")
    if uploaded_file:
        img_array = cv2.cvtColor(np.array(Image.open(uploaded_file).convert('RGB')), cv2.COLOR_RGB2BGR)
        
        c_img1, c_img2 = st.columns(2)
        with c_img1:
            st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c_img2:
            img_ph = st.empty()
            
        st.write("")
        col1, col2, col_div, col3, col4 = st.columns([2, 2, 0.2, 2, 2])
        dash_ph = [col1.empty(), col2.empty(), col_div.empty(), col3.empty(), col4.empty()]
        
        if st.button("🚀 بدء التحليل", key="btn_img"):
            out_rgb, dens, emp, m_r, w_r = analyze_pixels(img_array)
            img_ph.image(out_rgb, use_container_width=True)
            update_ui(dens, emp, m_r, w_r, dash_ph)

with tab2:
    uploaded_video = st.file_uploader("📂 ارفع مقطع فيديو (MP4)...", type=["mp4"], key="vid_upload")
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        img_ph = st.empty()
        
        st.write("")
        col1, col2, col_div, col3, col4 = st.columns([2, 2, 0.2, 2, 2])
        dash_ph = [col1.empty(), col2.empty(), col_div.empty(), col3.empty(), col4.empty()]
        
        if st.button("🚀 تشغيل الأرصاد", key="btn_vid"):
            cap = cv2.VideoCapture(tfile.name)
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                count += 1
                if count % 3 != 0: continue
                    
                out_rgb, dens, emp, m_r, w_r = analyze_pixels(frame)
                img_ph.image(out_rgb, use_container_width=True)
                update_ui(dens, emp, m_r, w_r, dash_ph)
            cap.release()

with tab3:
    youtube_url = st.text_input("🔗 أدخل رابط يوتيوب:", key="yt_url")
    if youtube_url:
        img_ph = st.empty()
        
        st.write("")
        col1, col2, col_div, col3, col4 = st.columns([2, 2, 0.2, 2, 2])
        dash_ph = [col1.empty(), col2.empty(), col_div.empty(), col3.empty(), col4.empty()]
        
        if st.button("🚀 بدء الاستشعار الحي", key="btn_live"):
            try:
                ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    stream_url = info['url']
                
                cap = cv2.VideoCapture(stream_url)
                count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    count += 1
                    if count % 15 != 0: continue
                        
                    out_rgb, dens, emp, m_r, w_r = analyze_pixels(frame)
                    img_ph.image(out_rgb, use_container_width=True)
                    update_ui(dens, emp, m_r, w_r, dash_ph)
                cap.release()
            except:
                pass
