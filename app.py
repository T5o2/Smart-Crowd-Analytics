import streamlit as st
from PIL import Image

# 1. إعدادات الصفحة
st.set_page_config(page_title="أرصاد الحشود | Crowd Analytics", page_icon="🕋", layout="wide")

# 2. تصميم CSS احترافي للوحة قيادة الأرصاد (Dashboard)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    .block-container {padding-top: 2rem !important;}
    
    /* تصميم بطاقات الأرصاد لتكون فخمة ومميزة */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border-top: 6px solid #d4af37; /* لون ذهبي يتناسب مع هوية الحرم */
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: #1e1e1e;
            border-top: 6px solid #d4af37;
        }
    }
</style>
""", unsafe_allow_html=True)

# 3. القائمة الجانبية: اختيار مصدر البيانات
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3340/3340156.png", width=80)
    st.title("📡 مركز التحكم")
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

# 6. لوحة أرصاد الزحام (تصميم مبدئي ليراه المستخدم كنتيجة نهائية)
st.markdown("### 🌡️ مؤشرات أرصاد الحشود (لوحة القيادة)")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('''
        <div class="metric-card">
            <h4 style="color:gray;">مؤشر الكثافة (Density)</h4>
            <h1 style="color:#ff4b4b;">85%</h1>
            <p>🔴 زحام شديد (حرجة)</p>
        </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
        <div class="metric-card">
            <h4 style="color:gray;">توزيع الفئات (استنتاج لوني)</h4>
            <h2>⚪ 65% رجال (إحرام)</h2>
            <h2>⚫ 35% نساء (عباءات)</h2>
        </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
        <div class="metric-card">
            <h4 style="color:gray;">حالة الموقع والتوصية</h4>
            <h2 style="color:#ffa500;">⚠️ يتطلب تدخل</h2>
            <p>ينصح بتوجيه تدفق المعتمرين إلى أدوار التوسعة لتخفيف الضغط عن صحن المطاف.</p>
        </div>
    ''', unsafe_allow_html=True)
