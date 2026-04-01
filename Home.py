import streamlit as st

st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 ระบบทำนายโรคพืชมะเขือเทศ")
st.markdown("### ด้วย Machine Learning และ Deep Learning")

st.divider()

# ===== บทนำ =====
st.header("โปรเจกต์นี้คืออะไร?")
st.markdown("""
ระบบทำนายโรคพืชมะเขือเทศ เป็นโครงการที่ผสมผสาน **เทคโนโลยี AI สองวิธี** เพื่อช่วยให้เกษตรกรและผู้เชี่ยวชาญ
สามารถตรวจจับโรคใบมะเขือเทศได้อย่างแม่นยำและรวดเร็ว

**ความสำคัญ:**
- โรคใบพืชเป็นปัญหาใหญ่ที่ทำให้ผลผลิตลดลง 20-40% 
- การตรวจจับเร็วช่วยบันทึกพืช ลดการแพร่ของโรค
- ระบบ AI ช่วยให้ตัดสินใจได้อย่างเป็นวิทยาศาสตร์
""")

st.divider()

# ===== เปรียบเทียบโมเดล =====
st.header("วิธีการทำนายสองแบบ")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model 1: Ensemble Learning")
    st.markdown("""
    **วิธีการ:** Random Forest + SVM + XGBoost
    
    **ข้อมูลที่ใช้:**
    - 🌡️ อุณหภูมิ (Temperature)
    - 💧 ความชื้น (Humidity)
    - 🌧️ ปริมาณฝน (Rainfall)
    - ⚗️ pH ของดิน (Soil pH)
    
    **ผลลัพธ์:**
    - ✅ ความแม่นยำ: **83.07%**
    - ⚡ ความเร็ว: **เร็วมาก** (โหลดข้อมูล < 1 วินาที)
    - 📈 ขนาดโมเดล: เล็ก (~5MB)
    
    **ข้อดี:**
    - เร็วและใช้ทรัพยากรน้อย
    - เหมาะสำหรับการใช้งานจริง
    
    **ข้อจำกัด:**
    - ต้องวัดพารามิเตอร์สิ่งแวดล้อม
    - ความแม่นยำ: ปานกลาง
    """)

with col2:
    st.subheader("🖼️ Model 2: Deep Learning (MobileNetV2)")
    st.markdown("""
    **วิธีการ:** Convolutional Neural Network (CNN)
    
    **ข้อมูลที่ใช้:**
    - 📸 ภาพใบมะเขือเทศ (224×224 RGB)
    - ตรวจจับจากรูปภาพเท่านั้น
    
    **ผลลัพธ์:**
    - ✅ ความแม่นยำ: **88.97%**
    - ⚡ ความเร็ว: **ปานกลาง** (~ 2-3 วินาที)
    - 📈 ขนาดโมเดล: ใหญ่ (~50MB)
    
    **ข้อดี:**
    - ความแม่นยำสูงกว่า
    - ใช้งานง่าย (ถ่ายรูปเท่านั้น)
    - ตรวจจับรายละเอียดเล็ก ๆ
    
    **ข้อจำกัด:**
    - ต้องมีภาพคุณภาพดี
    - ใช้ทรัพยากร CPU/GPU มากกว่า
    """)

st.divider()

# ===== โรคที่ตรวจจับได้ =====
st.header("โรคมะเขือเทศ 10 ประเภทที่ตรวจจับความ")
col1, col2, col3 = st.columns(3)

diseases = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Two-spotted Spider Mite",
    "Healthy (ปกติ)"
]

with col1:
    for disease in diseases[0:4]:
        st.markdown(f"- {disease}")

with col2:
    for disease in diseases[4:7]:
        st.markdown(f"- {disease}")

with col3:
    for disease in diseases[7:]:
        st.markdown(f"- {disease}")

st.divider()

# ===== วิธีการใช้งาน =====
st.header("วิธีการใช้งาน")

st.markdown("""
### 1️⃣ เข้าสู่หน้าอธิบาย (Explanation)
- เรียนรู้ทฤษฎีเกี่ยวกับโมเดล
- ดูกราฟผลลัพธ์และการเปรียบเทียบ
- เข้าใจวิธีการทำงานของแต่ละโมเดล

### 2️⃣ ทดสอบโมเดล (Test)
- **Model 1 Test**: กรอกข้อมูลสิ่งแวดล้อม หรือเลือกจากชุดข้อมูลทดสอบ
- **Model 2 Test**: อัปโหลดรูปใบมะเขือเทศ หรือเลือกตัวอย่างจากชุดข้อมูล
""")

st.divider()

# ===== คำแนะนำ =====
st.header("💡 คำแนะนำการใช้งาน")

tip1, tip2 = st.columns(2)

with tip1:
    st.info("""
    **สำหรับ Model 1:**
    ✓ ใช้ข้อมูลจากอุณหภูมิ ความชื้น ฯลฯ
    ✓ เหมาะสำหรับการบ้าน/สำนักงาน
    ✓ ได้ผลลัพธ์ทันทีประเมิน
    ✓ ไม่ต้องรูปภาพ
    """)

with tip2:
    st.info("""
    **สำหรับ Model 2:**
    ✓ ถ่ายรูปใบมะเขือเทศที่สดใจ
    ✓ ให้แสงสว่างที่ดี
    ✓ ถ่ายให้ชัดเจน เห็นลักษณ์โรณ
    ✓ ความแม่นยำขึ้นอยู่กับคุณภาพภาพ
    """)

st.divider()

# ===== เนวิเกชัน =====
st.header("🚀 เริ่มต้น")
st.markdown("""
👈 **เลือกหน้าจาก Sidebar ด้านซ้าย**

- ไปที่ **1_Model1_Explanation** หรือ **2_Model2_Explanation** เพื่อเรียนรู้
- ไปที่ **3_Model1_Test** หรือ **4_Model2_Test** เพื่อทดสอบ
""")
