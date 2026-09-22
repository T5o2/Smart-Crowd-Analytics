import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="أرصاد الحشود | Crowd Analytics", page_icon="🕋", layout="wide")

# 1. تنظيف الواجهة بدون إجبار (RTL) الذي يخرب التنسيق
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem !important;
    }
    
    .metric-box {
        background-color: #1e1e2f;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-left: 5px solid #d4af37;
        margin-bottom: 15px;
    }
    .metric-title { color: #a0a0b0; font-size: 1.1rem; margin-bottom: 10px; }
    .metric-val { font-size: 2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 2. تحميل نموذج YOLO (للبحث عن البشر فقط - كلاس 0)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 3. القائمة الجانبية الأنيقة (نفس مشروعك القديم)
col_settings, col_empty = st.columns([1, 10])
with col_settings:
    with st.popover("⚙️ إعدادات النظام"):
        source_type = st.radio("مصدر البيانات:", ["صورة ثابتة", "فيديو مسجل", "يوتيوب مباشر"], label_visibility="collapsed")
        conf_thresh = st.slider("دقة الرصد:", 0.1, 1.0, 0.25, 0.05)

st.title("🕋 المنصة الذكية لأرصاد الحشود")
st.markdown("### تحليل الكثافة وتوزيع الفئات باستخدام (YOLO + Color AI)")
st.markdown("---")

# 4. محرك الذكاء الاصطناعي المزدوج
def analyze_crowd_smart(image_array, conf):
    # 1. رصد البشر باستخدام YOLO
    results = model.predict(image_array, classes=[0], conf=conf)
    
    img_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    men_count = 0
    women_count = 0
    total_people = len(results[0].boxes)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # اقتطاع صورة الشخص فقط من داخل المربع
        person_crop = img_cv2[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            continue
            
        hsv_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        
        # فحص اللون الأبيض (إحرام) داخل مربع الشخص فقط
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv_crop, lower_white, upper_white)
        
        # فحص الألوان الداكنة (عباءات) داخل مربع الشخص فقط
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])
        mask_black = cv2.inRange(hsv_crop, lower_black, upper_black)
        
        white_px = cv2.countNonZero(mask_white)
        black_px = cv2.countNonZero(mask_black)
        
        # تصنيف الشخص ورسم المربع
        if white_px > black_px:
            men_count += 1
            color = (255, 255, 255) # مربع أبيض للرجل
            label = "Ihram"
        else:
            women_count += 1
            color = (255, 0, 0) # مربع أزرق للمرأة (للتوضيح)
            label = "Abaya"
            
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        
    res_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    
    # حساب نسبة الزحام (بافتراض أن 1000 شخص في الكادر يعني زحام 100%)
    MAX_CAPACITY = 1000
    density = min(int((total_people / MAX_CAPACITY) * 100), 100)
    
    # حساب النسب المئوية
    if total_people > 0:
        men_ratio = int((men_count / total_people) * 100)
        women_ratio = 100 - men_ratio
    else:
        men_ratio = 0
        women_ratio = 0
        
    return density, total_people, men_ratio, women_ratio, res_rgb

# 5. معالجة الأقسام
if source_type == "صورة ثابتة":
    uploaded_file = st.file_uploader("📂 ارفع صورة لساحات الحرم...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 اللقطة الأصلية")
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("#### 🎯 الرصد الآلي (YOLO + Colors)")
            res_placeholder = st.empty()
            
        if st.button("🚀 بدء تحليل الزحام"):
            with st.spinner("جاري رصد الأشخاص وتحليل الألوان..."):
                img_array = np.array(image)
                density, total_ppl, men_r, women_r, out_img = analyze_crowd_smart(img_array, conf_thresh)
                
                res_placeholder.image(out_img, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 📊 تقرير أرصاد الحشود الفعلي")
                st.progress(density / 100)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f'<div class="metric-box"><div class="metric-title">مؤشر الكثافة</div><div class="metric-val" style="color:{"#ff4b4b" if density > 70 else "#00fa9a"}">{density}%</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-box"><div class="metric-title">الأشخاص المرصودين</div><div class="metric-val" style="color:#fff">{total_ppl}</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-box"><div class="metric-title">رجال (إحرام)</div><div class="metric-val" style="color:#fff">{men_r}%</div></div>', unsafe_allow_html=True)
                c4.markdown(f'<div class="metric-box"><div class="metric-title">نساء (عباءات)</div><div class="metric-val" style="color:#4da6ff">{women_r}%</div></div>', unsafe_allow_html=True)

elif source_type == "فيديو مسجل":
    st.info("تم فتح القسم! يمكنك رفع فيديو قصير، ولكن معالجة الفيديو إطاراً بإطار تتطلب وقت حوسبة طويل على المتصفح.")
    uploaded_video = st.file_uploader("📂 ارفع مقطع فيديو (MP4)...", type=["mp4"])
    if uploaded_video:
        st.video(uploaded_video)

elif source_type == "يوتيوب مباشر":
    st.info("القسم مفتوح. أضف رابط البث:")
    youtube_url = st.text_input("🔗 رابط YouTube:")
    if youtube_url:
        st.video(youtube_url)
