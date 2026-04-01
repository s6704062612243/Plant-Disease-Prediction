import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Model 1 Test", page_icon="📊", layout="wide")

st.title("Model 1: Test Prediction")
st.write("ทำนายว่าพืชมีโรคหรือไม่ จากข้อมูลเชิงตาราง")
st.divider()

# ===== Load Model =====
@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, "model1_final.pkl")
        
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            return None
    except ImportError as e:
        if 'xgboost' in str(e) or 'xgb' in str(e).lower():
            return None
        else:
            raise
    except Exception as e:
        return None

# ===== Load Test Data =====
@st.cache_data
def load_test_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        test_file = os.path.join(parent_dir, "models", "model1_test_samples.csv")
        
        if os.path.exists(test_file):
            df = pd.read_csv(test_file)
            df = df.dropna()
            return df
        else:
            # สร้าง sample data ถ้าไม่มีไฟล์
            return create_sample_data()
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถโหลดข้อมูลทดสอบ: {str(e)}")
        return create_sample_data()

# ===== Create Sample Data =====
def create_sample_data():
    sample_data = {
        'temperature': [15.5, 22.3, 28.7, 35.2, 24.1, 18.9, 32.5, 20.0, 27.3, 26.8],
        'humidity': [45.2, 62.5, 71.3, 88.0, 55.0, 48.7, 85.2, 50.1, 68.5, 72.0],
        'rainfall': [5.2, 15.3, 25.5, 55.0, 12.1, 8.5, 45.2, 10.0, 20.3, 22.5],
        'soil_pH': [6.2, 6.8, 7.1, 7.5, 6.5, 6.1, 7.3, 6.0, 6.9, 7.0],
        'actual_label': [0, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    }
    return pd.DataFrame(sample_data)

# ===== Mock Prediction Function =====
def mock_predict(sample):
    """ทำนายผล (ใช้เมื่อไม่มีโมเดล)"""
    temp, humid, rain, ph = sample[0]
    score = (temp - 25)**2 / 100 + (humid - 70) / 100 + (rain - 20) / 100
    if score > 3:
        return np.array([1])
    else:
        return np.array([0])

# ===== Load Resources =====
model = load_model()
test_df = load_test_data()

# ===== Tab Interface =====
tab1, tab2 = st.tabs(["ป้อนข้อมูลเอง", "ทดสอบจากชุดข้อมูล"])

with tab1:
    st.header("กรอกข้อมูลสภาพแวดล้อม")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temp = st.slider("อุณหภูมิ (°C)", 0.0, 60.0, 25.0, step=0.5)
        humid = st.slider("ความชื้น (%)", 0.0, 100.0, 70.0, step=1.0)
    
    with col2:
        rain = st.slider("ปริมาณฝน (mm)", 0.0, 100.0, 10.0, step=1.0)
        ph = st.slider("pH ของดิน", 0.0, 14.0, 6.5, step=0.1)
    
    st.divider()
    
    if st.button("ทำนายผล", use_container_width=True, key="predict_manual"):
        sample = np.array([[temp, humid, rain, ph]])
        
        if model is not None:
            pred = model.predict(sample)[0]
        else:
            pred = mock_predict(sample)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ผลการทำนาย", "มีโรค" if pred == 1 else "ไม่มีโรค")
        with col2:
            status = "แบบจำลอง Demo" if model is None else "แบบจำลองจริง"
            st.metric("สถานะ", status)

with tab2:
    st.header("ทดสอบจากชุดข้อมูล")
    
    if test_df.empty:
        st.info("ไม่พบข้อมูล")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            idx = st.select_slider(
                "เลือกตัวอย่างข้อมูล",
                options=range(len(test_df)),
                value=0
            )
        
        selected = test_df.iloc[idx]
        
        # Display metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("อุณหภูมิ", f"{selected['temperature']:.1f}°C")
        with metric_col2:
            st.metric("ความชื้น", f"{selected['humidity']:.1f}%")
        with metric_col3:
            st.metric("ปริมาณฝน", f"{selected['rainfall']:.1f}mm")
        with metric_col4:
            st.metric("pH", f"{selected['soil_pH']:.2f}")
        
        st.divider()
        
        if st.button("ทำนายผล", use_container_width=True, key="predict_dataset"):
            sample = np.array([[
                selected['temperature'],
                selected['humidity'],
                selected['rainfall'],
                selected['soil_pH']
            ]])
            
            if model is not None:
                pred = model.predict(sample)[0]
            else:
                pred = mock_predict(sample)[0]
            
            actual = int(selected['actual_label'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                pred_text = "มีโรค" if pred == 1 else "ไม่มีโรค"
                st.metric("ผลการทำนาย", pred_text)
            with col2:
                actual_text = "มีโรค" if actual == 1 else "ไม่มีโรค"
                st.metric("ผลจริง", actual_text)
            with col3:
                if pred == actual:
                    st.metric("ความถูกต้อง", "ถูกต้อง")
                else:
                    st.metric("ความถูกต้อง", "ผิดพลาด")
