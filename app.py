import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import yt_dlp

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
        border-bottom: 4px solid #d4af37;
    }
    .metric-title { color: #a0a0b0; font-size: 1.1rem; margin-bottom: 10px; }
    .metric-val { font-size: 2.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# القائمة الجانبية المصغرة والأنيقة
col_settings, col_empty = st.columns([1, 10])
with col_settings:
    with st.popover("⚙️ إعدادات النظام"):
        source_type = st.radio("مصدر البيانات:", ["صورة ثابتة", "فيديو مسجل", "بث مباشر (يوتيوب)"], label_visibility="collapsed")

st.title("🕋 المنصة الذكية لأرصاد الحشود")
st.markdown("### نظام الاستشعار الموحد (Blob Analysis) لتقدير الكثافة والنسب")
st.markdown("---")

# ---------------------------------------------------------
# محرك الذكاء الاصطناعي الجديد (تحليل الكتل بدلاً من المربعات)
# ---------------------------------------------------------
def process_crowd_cv(frame_bgr):
    # تحويل الصورة إلى نطاق HSV لفرز الألوان بدقة
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
    # 1. التقاط اللون الأبيض (الإحرام)
    lower_white = np.array([0, 0, 190])
    upper_white = np.array([180, 40, 255])
    mask_w = cv2.inRange(hsv, lower_white, upper_white)
    
    # 2. التقاط اللون الأسود/الداكن (العباءات)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])
    mask_b = cv2.inRange(hsv, lower_black, upper_black)
    
    # استخراج الكتل (Contours)
    contours_w, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_w_area = 0
    valid_b_area = 0
    output_overlay = frame_bgr.copy()
    
    # تصفية الكتل البيضاء (الرجال) - تجاهل الرخام الضخم والضجيج الصغير
    for cnt in contours_w:
        area = cv2.contourArea(cnt)
        if 2 < area < 4000:  # هذا الشرط العبقري يمنع رصد الرخام
            valid_w_area += area
            cv2.drawContours(output_overlay, [cnt], -1, (255, 255, 255), -1) # تلوين الكتلة بأبيض ساطع
            
    # تصفية الكتل الداكنة (النساء) - تجاهل الكعبة الضخمة والظلال الكبيرة
    for cnt in contours_b:
        area = cv2.contourArea(cnt)
        if 2 < area < 5000: # هذا الشرط يمنع رصد الكعبة
            valid_b_area += area
            cv2.drawContours(output_overlay, [cnt], -1, (200, 150, 0), -1) # تلوين الكتلة بأزرق لتمييزها بصرياً
            
    # دمج الخريطة اللونية مع الصورة الأصلية
    final_frame = cv2.addWeighted(frame_bgr, 0.7, output_overlay, 0.3, 0)
    final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
    
    # حساب الكثافة (Density Math)
    total_pixels = frame_bgr.shape[0] * frame_bgr.shape[1]
    # نفترض أن امتلاء 20% من إجمالي بكسلات الشاشة بالبشر يعني زحام 100% (لأن الباقي مباني وسماء)
    max_crowd_area = total_pixels * 0.20
    total_valid_area = valid_w_area + valid_b_area
    
    density = min(int((total_valid_area / max_crowd_area) * 100), 100)
    
    # حساب النسب
    if total_valid_area > 0:
        men_ratio = int((valid_w_area / total_valid_area) * 100)
        women_ratio = 100 - men_ratio
    else:
        men_ratio, women_ratio = 0, 0
        
    return final_frame_rgb, density, men_ratio, women_ratio

# دالة لتحديث لوحة الأرصاد في الواجهة (تم حذف عداد الأشخاص)
def update_dashboard(density, men_r, women_r, placeholders):
    color_dens = "#ff4b4b" if density > 70 else "#00fa9a"
    placeholders[0].markdown(f'<div class="metric-box"><div class="metric-title">مؤشر الكثافة</div><div class="metric-val" style="color:{color_dens}">{density}%</div></div>', unsafe_allow_html=True)
    placeholders[1].markdown(f'<div class="metric-box"><div class="metric-title">رجال (إحرام)</div><div class="metric-val" style="color:#fff">{men_r}%</div></div>', unsafe_allow_html=True)
    placeholders[2].markdown(f'<div class="metric-box"><div class="metric-title">نساء (عباءات)</div><div class="metric-val" style="color:#4da6ff">{women_r}%</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# توجيه المدخلات
# ---------------------------------------------------------
if source_type == "صورة ثابتة":
    uploaded_file = st.file_uploader("📂 ارفع صورة لساحات الحرم...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_array_rgb = np.array(image)
        img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 اللقطة الأصلية")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("#### 🎯 خريطة الرصد اللحظي")
            img_placeholder = st.empty()
            
        st.markdown("---")
        st.markdown("#### 📊 لوحة الأرصاد")
        # تقسيم اللوحة إلى 3 مربعات فقط
        c1, c2, c3 = st.columns(3)
        dash_placeholders = [c1.empty(), c2.empty(), c3.empty()]
        
        if st.button("🚀 بدء تحليل الزحام"):
            with st.spinner("جاري تحليل الكتل اللونية..."):
                out_rgb, dens, m_r, w_r = process_crowd_cv(img_array_bgr)
                img_placeholder.image(out_rgb, use_container_width=True)
                update_dashboard(dens, m_r, w_r, dash_placeholders)

elif source_type == "فيديو مسجل":
    uploaded_video = st.file_uploader("📂 ارفع مقطع فيديو (MP4)...", type=["mp4", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        st.markdown("#### 🎯 الاستشعار الحركي اللحظي")
        img_placeholder = st.empty()
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        dash_placeholders = [c1.empty(), c2.empty(), c3.empty()]
        
        stop_btn = st.button("⏹️ إيقاف التحليل")
        
        if not stop_btn:
            cap = cv2.VideoCapture(tfile.name)
            frame_skip = 5 
            count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                count += 1
                if count % frame_skip != 0: continue
                    
                out_rgb, dens, m_r, w_r = process_crowd_cv(frame)
                img_placeholder.image(out_rgb, use_container_width=True)
                update_dashboard(dens, m_r, w_r, dash_placeholders)
                
            cap.release()

elif source_type == "بث مباشر (يوتيوب)":
    youtube_url = st.text_input("🔗 أدخل رابط بث يوتيوب مباشر:")
    if youtube_url:
        st.markdown("#### 🔴 الاستشعار من البث المباشر")
        img_placeholder = st.empty()
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        dash_placeholders = [c1.empty(), c2.empty(), c3.empty()]
        
        stop_btn = st.button("⏹️ إيقاف البث")
        
        if not stop_btn:
            try:
                ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    stream_url = info['url']
                
                cap = cv2.VideoCapture(stream_url)
                frame_skip = 30 
                count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    count += 1
                    if count % frame_skip != 0: continue
                        
                    out_rgb, dens, m_r, w_r = process_crowd_cv(frame)
                    img_placeholder.image(out_rgb, use_container_width=True)
                    update_dashboard(dens, m_r, w_r, dash_placeholders)
                    
                cap.release()
            except Exception as e:
                st.error(f"❌ حدث خطأ في سحب البث: {e}")
