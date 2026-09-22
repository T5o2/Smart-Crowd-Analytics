import streamlit as st
from PIL import Image

# 1. إعدادات الصفحة
st.set_page_config(page_title="أرصاد الحشود | Crowd Analytics", page_icon="🕋", layout="wide")

# 2. تصميم CSS احترافي للوحة القيادة وفرض اتجاه اللغة العربية (RTL)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    
    /* قلب الموقع بالكامل ليدعم العربية من اليمين لليسار */
    .block-container {
        padding-top: 2rem !important;
        direction: rtl;
    }
    
    /* تعديل القائمة الجانبية لتكون RTL */
    [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }
    
    /* تصميم بطاقات الأرصاد لتكون فخمة ومميزة */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 25px 20px;
        text-align: center;
        border-top: 6px solid #d4af37; /* لون ذهبي */
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-title {
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 15px;
        font-weight: 600;
    }
    .metric-value-red {
        color: #ff4b4b;
        font-size: 3rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-value-gold {
        color: #d4af37;
        font-size: 2rem;
        font-weight: bold;
        margin: 15px 0;
    }
    .metric-text {
        font-size: 1.3rem;
        font-weight: bold;
        color: #333;
    }
    .metric-subtext {
        font-size: 0.95rem;
        color: #666;
        line-height: 1.5;
        margin-top: 10px;
    }
    
    @media (prefers-color-scheme: dark) {
        .metric-card { background-color: #1e1e1e; }
        .metric-text { color: #eee; }
        .metric-subtext { color: #aaa; }
    }
</style>
""", unsafe_allow_html=True)

# 3. القائمة الجانبية: اختيار مصدر البيانات
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>📡</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>مركز التحكم</h2>", unsafe_allow_html=True)
    st.markdown("---")
    source_type = st.radio("📍 حدد مصدر البيانات للتحليل:", 
                           ["📷 تحليل صورة ثابتة", "🎥 تحليل فيديو مسجل", "🔗 استشعار بث يوتيوب مباشر"])
    st.markdown("---")
    st.info("💡 يعتمد هذا النظام على خرائط الكثافة (Density Maps) والتجزئة اللونية (Color Segmentation) لتحليل الحشود عن بُعد.")

# 4. الواجهة الرئيسية
st.title("🕋 المنصة الذكية لأرصاد الحشود")
st.markdown("### نظام استشعار لحظي لتقدير الكثافة وتوزيع المعتمرين")
st.markdown("---")

# 5. منطق التعامل مع مدخلات الباحث
if source_type == "📷 تحليل صورة ثابتة":
    uploaded_file = st.file_uploader("📂 ارفع صورة لساحات الحرم أو موقع الزحام (JPG, PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True, caption="الصورة المدخلة")
        if st.button("🚀 بدء تحليل الزحام"):
            st.warning("⏳ سيتم ربط محرك الذكاء الاصطناعي (OpenCV) هنا في الخطوة القادمة!")

elif source_type == "🎥 تحليل فيديو مسجل":
    uploaded_video = st.file_uploader("📂 ارفع مقطع فيديو (MP4)", type=["mp4"])
    if uploaded_video:
        st.video(uploaded_video)
        if st.button("🚀 بدء التحليل الحركي"):
            st.warning("⏳ سيتم ربط خوارزمية تتبع الإطارات هنا في الخطوة القادمة!")

elif source_type == "🔗 استشعار بث يوتيوب مباشر":
    youtube_url = st.text_input("🔗 أدخل رابط البث المباشر (YouTube):", placeholder="مثال: https://www.youtube.com/watch?v=...")
    if youtube_url:
        st.video(youtube_url)
        if st.button("🚀 بدء الاستشعار اللحظي"):
            st.warning("⏳ سيتم دمج خوارزمية (yt-dlp) لسحب البث وتحليله هنا في الخطوة القادمة!")

st.markdown("---")

# 6. لوحة أرصاد الزحام (التصميم الاحترافي الجديد)
st.markdown("### 🌡️ مؤشرات أرصاد الحشود (لوحة القيادة)")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('''
        <div class="metric-card">
            <div class="metric-title">مؤشر الكثافة (Density)</div>
            <div class="metric-value-red">85%</div>
            <div class="metric-text">🔴 زحام شديد (حرجة)</div>
        </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
        <div class="metric-card">
            <div class="metric-title">توزيع الفئات (استنتاج لوني)</div>
            <div class="metric-text" style="margin-top: 15px;">⚪ 65% رجال (إحرام)</div>
            <div class="metric-text" style="margin-top: 15px;">⚫ 35% نساء (عباءات)</div>
        </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
        <div class="metric-card">
            <div class="metric-title">حالة الموقع والتوصية</div>
            <div class="metric-value-gold">⚠️ يتطلب تدخل</div>
            <div class="metric-subtext">ينصح بتوجيه تدفق المعتمرين إلى أدوار التوسعة لتخفيف الضغط عن صحن المطاف.</div>
        </div>
    ''', unsafe_allow_html=True)
