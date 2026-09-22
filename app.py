import streamlit as st
from PIL import Image

# 1. إعدادات الصفحة
st.set_page_config(page_title="أرصاد الحشود | Crowd Analytics", page_icon="🕋", layout="wide")

# 2. تنظيف الواجهة برمجياً (نفس أكواد المشروع السابق بالضبط)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    
    [data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 5px solid #d4af37; /* لون ذهبي يتناسب مع الحرم */
    }
    @media (prefers-color-scheme: dark) {
        [data-testid="stMetric"] {
            background-color: #1e1e1e;
            border-left: 5px solid #d4af37;
        }
    }
    
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. لوحة الإعدادات العائمة الأنيقة (نفس الميزة في مشروعك السابق)
col_settings, col_empty = st.columns([1, 10])

with col_settings:
    with st.popover("⚙️ إعدادات النظام", help="اضغط لتغيير مصدر البيانات"):
        st.markdown("**🌐 مصدر البيانات**")
        source_type = st.radio("حدد المصدر:", ["صورة ثابتة", "فيديو مسجل", "بث مباشر (يوتيوب)"], label_visibility="collapsed")

# 4. بناء الواجهة الرئيسية
st.title("🕋 المنصة الذكية لأرصاد الحشود")
st.markdown("### نظام استشعار لحظي لتقدير الكثافة وتوزيع المعتمرين")
st.markdown("---")

# 5. منطقة الإدخال والتحليل
if source_type == "صورة ثابتة":
    uploaded_file = st.file_uploader("🤖 ارفع صورة لساحات الحرم (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 الصورة الأصلية")
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("#### 🎯 خريطة الكثافة الآلية")
            st.info("⏳ سيتم عرض خريطة الكثافة (Density Map) هنا بعد ربط محرك الذكاء الاصطناعي.")
            
        if st.button("🚀 بدء الفحص"):
            st.warning("جاري برمجة خوارزمية الذكاء الاصطناعي...")

elif source_type == "فيديو مسجل":
    uploaded_video = st.file_uploader("📂 ارفع مقطع فيديو (MP4)...", type=["mp4"])
    if uploaded_video is not None:
        st.video(uploaded_video)

elif source_type == "بث مباشر (يوتيوب)":
    youtube_url = st.text_input("🔗 أدخل رابط البث المباشر (YouTube):")
    if youtube_url:
        st.video(youtube_url)

st.markdown("---")

# 6. لوحة الإحصائيات (بأسلوب st.metric المعتمد في مشروعك)
st.markdown("### 📊 تقرير أرصاد الحشود اللحظي")

# أرقام وهمية مؤقتة لعرض التصميم
density_score = 85

st.markdown(f"#### 📈 مؤشر الكثافة : {density_score}%")
st.progress(density_score / 100)

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
stat_col1.metric(label="🔴 مؤشر الزحام", value=f"{density_score}%")
stat_col2.metric(label="⚪ نسبة الرجال (إحرام)", value="65%")
stat_col3.metric(label="⚫ نسبة النساء (عباءات)", value="35%")
stat_col4.metric(label="⚠️ حالة الموقع", value="حرجة")

if density_score > 80:
    st.error("⚠️ **تنبيه:** زحام شديد! يُنصح بتوجيه تدفق المعتمرين إلى أدوار التوسعة وتخفيف الضغط عن صحن المطاف.")
else:
    st.success("✅ **الحالة مستقرة:** انسيابية في الحركة.")
