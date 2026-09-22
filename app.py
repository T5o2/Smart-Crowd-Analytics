import streamlit as st
from PIL import Image
import cv2
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="أرصاد الحشود | Crowd Analytics", page_icon="🕋", layout="wide")

# 2. تصميم CSS عصري واحترافي (وداعاً للتصميم البيسك)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem !important;
        direction: rtl;
    }
    
    /* تصميم بطاقات المؤشرات المتقدمة */
    .custom-card {
        background-color: #1e1e2f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 15px;
        border: 1px solid #333;
    }
    .custom-card h4 {
        color: #a0a0b0;
        font-size: 1.1rem;
        margin-bottom: 15px;
        font-weight: 500;
    }
    .value-red { color: #ff4b4b; font-size: 2.5rem; font-weight: bold; }
    .value-green { color: #00fa9a; font-size: 2.5rem; font-weight: bold; }
    .value-blue { color: #4da6ff; font-size: 2rem; font-weight: bold; margin-bottom: 5px;}
    .value-white { color: #ffffff; font-size: 2rem; font-weight: bold; }
    
    .status-alert { color: #ff4b4b; font-size: 1.2rem; font-weight: bold; margin-top: 10px;}
    .status-safe { color: #00fa9a; font-size: 1.2rem; font-weight: bold; margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

# 3. محرك الذكاء الاصطناعي (تحليل الكثافة والألوان)
def analyze_crowd_heuristic(image_array):
    # تحويل الصورة إلى وضع HSV الأفضل في التقاط الألوان
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    # 1. التقاط اللون الأبيض (المحرمين)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # 2. التقاط الألوان الداكنة جداً (العباءات)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # حساب عدد البكسلات
    white_pixels = cv2.countNonZero(mask_white)
    black_pixels = cv2.countNonZero(mask_black)
    total_pixels = image_array.shape[0] * image_array.shape[1]
    
    # حساب إجمالي البشر التقديري
    crowd_pixels = white_pixels + black_pixels
    
    # تجنب القسمة على صفر إذا كانت الصورة فارغة
    if crowd_pixels == 0:
        return 0, 50, 50, image_array
        
    # معادلة الكثافة (نعتبر أن 30% من الشاشة لو امتلأت بالبشر تعني زحام 100%)
    density = min(int((crowd_pixels / (total_pixels * 0.30)) * 100), 100)
    
    # معادلة نسبة الرجال للنساء
    men_ratio = int((white_pixels / crowd_pixels) * 100)
    women_ratio = 100 - men_ratio
    
    # إنشاء "خريطة حرارية" بصرية (Heatmap) للنتيجة
    combined_mask = cv2.add(mask_white, mask_black)
    heatmap = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
    output_image = cv2.addWeighted(image_array, 0.7, heatmap, 0.3, 0)
    
    return density, men_ratio, women_ratio, output_image

# 4. لوحة الإعدادات العائمة
col_settings, col_empty = st.columns([1, 10])
with col_settings:
    with st.popover("⚙️ إعدادات النظام"):
        source_type = st.radio("مصدر البيانات:", ["صورة ثابتة", "فيديو مسجل", "يوتيوب مباشر"], label_visibility="collapsed")

# 5. الواجهة الرئيسية
st.title("🕋 المنصة الذكية لأرصاد الحشود")
st.markdown("### تحليل الكثافة وتوزيع الفئات باستخدام (Color Segmentation)")
st.markdown("---")

if source_type == "صورة ثابتة":
    uploaded_file = st.file_uploader("📂 ارفع صورة لساحات الحرم (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 اللقطة الأصلية")
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("#### 🎯 الخريطة الحرارية (Heatmap)")
            heatmap_placeholder = st.empty()
            heatmap_placeholder.info("اضغط على زر الفحص لتوليد الخريطة...")
            
        if st.button("🚀 بدء تحليل الزحام الفعلي"):
            with st.spinner("جاري مسح البكسلات وتحليل الألوان..."):
                # استدعاء محرك الذكاء الاصطناعي الحقيقي
                density, men, women, out_img = analyze_crowd_heuristic(img_array)
                
                # عرض الصورة المعالجة
                heatmap_placeholder.image(out_img, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 📊 لوحة أرصاد الحشود (نتائج حقيقية)")
                
                # تصميم البطاقات الديناميكي
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    density_color = "value-red" if density > 75 else "value-green"
                    density_status = "🔴 زحام شديد (حرجة)" if density > 75 else "🟢 انسيابية في الحركة"
                    status_class = "status-alert" if density > 75 else "status-safe"
                    
                    st.markdown(f'''
                        <div class="custom-card">
                            <h4>مؤشر الكثافة (Density)</h4>
                            <div class="{density_color}">{density}%</div>
                            <div class="{status_class}">{density_status}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                with c2:
                    st.markdown(f'''
                        <div class="custom-card">
                            <h4>توزيع الفئات (استنتاج لوني)</h4>
                            <div class="value-white">⚪ {men}% إحرام</div>
                            <div class="value-blue">⚫ {women}% عباءات</div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                with c3:
                    action_title = "⚠️ يتطلب تدخل" if density > 75 else "✅ الوضع آمن"
                    action_color = "#ff4b4b" if density > 75 else "#00fa9a"
                    action_desc = "توجيه المعتمرين لأدوار التوسعة فوراً." if density > 75 else "لا يوجد إجراء مطلوب حالياً."
                    
                    st.markdown(f'''
                        <div class="custom-card">
                            <h4>توصية النظام الآلية</h4>
                            <div style="color: {action_color}; font-size: 1.8rem; font-weight: bold; margin: 10px 0;">{action_title}</div>
                            <div style="color: #aaa; font-size: 0.95rem;">{action_desc}</div>
                        </div>
                    ''', unsafe_allow_html=True)

elif source_type in ["فيديو مسجل", "يوتيوب مباشر"]:
    st.info("قمنا بتعطيل هذا القسم مؤقتاً لنركز على اختبار الخوارزمية على الصور أولاً.")
