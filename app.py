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
    
    .stImage > img {
        max-height: 260px; 
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
        background: rgba(0, 0, 0, 0.75); 
        z-index: 1;
    }
    .metric-content {
        position: relative;
        z-index: 2;
    }
    
    .metric-title { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 1.1rem; font-weight: 600; margin-bottom: 5px; color: #ddd; }
    
    .metric-val { 
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 3rem; 
        font-weight: 900; 
        opacity: 0.85; 
        text-shadow: 0px 5px 15px rgba(0, 0, 0, 0.9), 0 0 12px currentColor; 
    }
    
    .divider-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 160px;
    }
    .divider-line {
        width: 3px;
        height: 70%;
        background: linear-gradient(to bottom, transparent, #d4af37, transparent); 
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.8);
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

def analyze_pixels(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h, w = frame.shape[:2]
    
    spatial_mask = np.ones((h, w), dtype=np.uint8) * 255
    cv2.ellipse(spatial_mask, (w//2, h//2), (int(w*0.12), int(h*0.15)), 0, 0, 360, 0, -1)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = cv2.convertScaleAbs(magnitude)
    texture_mask = cv2.inRange(magnitude, 40, 255)
    
    advanced_mask = cv2.bitwise_and(spatial_mask, texture_mask)
    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 50])
    
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    valid_white = cv2.bitwise_and(mask_white, advanced_mask)
    valid_dark = cv2.bitwise_and(mask_dark, advanced_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    valid_white = cv2.morphologyEx(valid_white, cv2.MORPH_OPEN, kernel, iterations=1)
    valid_dark = cv2.morphologyEx(valid_dark, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours_w, _ = cv2.findContours(valid_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_d, _ = cv2.findContours(valid_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_w = np.zeros_like(mask_white)
    final_d = np.zeros_like(mask_dark)
    
    total_px = h * w
    max_area = total_px * 0.01
    
    w_px, d_px = 0, 0
    
    for cnt in contours_w:
        area = cv2.contourArea(cnt)
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = float(cw)/ch if ch > 0 else 0
        
        if 45 < area < max_area and 0.3 < aspect_ratio < 3.0:
            cv2.drawContours(final_w, [cnt], -1, 255, -1)
            w_px += area
            
    for cnt in contours_d:
        area = cv2.contourArea(cnt)
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = float(cw)/ch if ch > 0 else 0
        
        if 45 < area < max_area and 0.3 < aspect_ratio < 3.0:
            cv2.drawContours(final_d, [cnt], -1, 255, -1)
            d_px += area
            
    roi_limit = total_px * 0.20
    total_crowd = w_px + d_px
    
    density = min(int((total_crowd / roi_limit) * 100), 100)
    empty = 100 - density
    
    if total_crowd > 0:
        m_r = int((w_px / total_crowd) * 100)
        w_r = 100 - m_r
    else:
        m_r, w_r = 0, 0
        
    overlay = frame.copy()
    overlay[final_w > 0] = [255, 255, 255]
    overlay[final_d > 0] = [255, 0, 255]
    
    res = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), density, empty, m_r, w_r

def generate_card(bg_img, title, val, color):
    return f'''
    <div class="metric-card" style="background-image: url('data:image/jpeg;base64,{bg_img}'); border-color: {color};">
        <div class="metric-content">
            <div class="metric-title">{title}</div>
            <div class="metric-val" style="color: {color};">{val}%</div>
        </div>
    </div>
    '''

def update_ui(density, empty, m_r, w_r, placeholders):
    card_red = generate_card(alert_bg, " مزدحم ", density, "#ff4b4b")
    card_green = generate_card(safe_bg, " فاضي ", empty, "#00fa9a")
    card_men = generate_card(men_bg, " الرجال ", m_r, "#ffffff")
    card_women = generate_card(women_bg, " النساء ", w_r, "#696969")
    
    if density >= empty:
        box1, box2 = card_red, card_green
    else:
        box1, box2 = card_green, card_red

    placeholders[0].markdown(box1, unsafe_allow_html=True)
    placeholders[1].markdown(box2, unsafe_allow_html=True)
    placeholders[2].markdown('<div class="divider-container"><div class="divider-line"></div></div>', unsafe_allow_html=True)
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
