import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model 1 Explanation", page_icon="📘", layout="wide")

st.title("Model 1: Ensemble Learning")
st.write("ศึกษา Dataset, Preprocessing และอัลกอริทึม Ensemble Learning")
st.divider()

st.header("1. Dataset")
st.write("""
- **ประเภท**: Tabular Data (CSV)
- **ขนาด**: ~3,000 แถว
- **ฟีเจอร์**: 4 ตัว (Temperature, Humidity, Rainfall, Soil pH)
- **เป้าหมาย**: Binary Classification (โรค/ไม่โรค)
""")

feature_data = pd.DataFrame({
    "ฟีเจอร์": ["temperature", "humidity", "rainfall", "soil_pH"],
    "ความหมาย": ["อุณหภูมิ (°C)", "ความชื้น (%)", "ปริมาณฝน (mm)", "ค่า pH"],
    "ช่วงค่า": ["-30 ถึง 60°C", "0 ถึง 100%", "0 ถึง ∞ mm", "0 ถึง 14"]
})
st.dataframe(feature_data, use_container_width=True)

st.header("2. Data Cleaning & Preprocessing")

st.subheader("2.1 ปัญหาของ Dataset (Dirty Data)")
st.write("""
ข้อมูลมีความไม่สมบูรณ์หลายประการ:
- **Missing Values**: ค่าบางตัวหายไป
- **Outliers**: ค่าผิดปกติ (เช่น อุณหภูมิ > 60°C หรือ < 0°C)
- **Duplicates**: แถวข้อมูลซ้ำ
- **Class Imbalance**: จำนวนโรค/ไม่โรค ไม่สมดุล
""")

st.subheader("2.2 ขั้นตอน Data Cleaning")
st.code("""
# 1) ลบข้อมูลซ้ำ
df_clean = df_dirty.drop_duplicates().copy()

# 2) แทนค่า Outliers เป็น NaN
df_clean.loc[(df_clean["temperature"] < 0) | (df_clean["temperature"] > 60), "temperature"] = np.nan
df_clean.loc[(df_clean["humidity"] < 0) | (df_clean["humidity"] > 100), "humidity"] = np.nan
df_clean.loc[(df_clean["rainfall"] < 0), "rainfall"] = np.nan
df_clean.loc[(df_clean["soil_pH"] < 0) | (df_clean["soil_pH"] > 14), "soil_pH"] = np.nan

# 3) เติมค่าที่หายไป ด้วย Median
for col in ["temperature", "humidity", "rainfall", "soil_pH"]:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# 4) แบ่ง Train/Test (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) สมดุล Class ด้วย SMOTETomek
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smt.fit_resample(X_train, y_train)
""", language="python")

st.header("3. Ensemble Methods")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Random Forest")
    st.write("- หลาย Decision Trees\n- ลงคะแนนส่วนใหญ่\n- ไม่ต้อง normalize")

with col2:
    st.subheader("SVM")
    st.write("- หาพื้นที่แยกคลาส\n- RBF kernel\n- จัดการ imbalance")

with col3:
    st.subheader("XGBoost")
    st.write("- Gradient Boosting\n- ต้นไม้แบบลำดับ\n- ประสิทธิภาพสูง")

st.code("""
# Soft Voting Ensemble
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('xgb', xgb)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
""", language="python")

st.header("4. Model Training & Building")

st.subheader("4.1 Baseline Model")
st.write("→ Random Forest เพียงอย่างเดียว (ใช้เป็นค่าเปรียบเทียบ)")

st.subheader("4.2 Ensemble Model (Final)")
st.code("""
# 3 Base Models
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
svm = SVC(kernel="rbf", C=2.0, probability=True, class_weight="balanced", random_state=42)
xgb = XGBClassifier(n_estimators=300, scale_pos_weight=scale_pos_weight, random_state=42)

# Soft Voting Ensemble
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('xgb', xgb)],
    voting='soft',
    weights=[1, 1.5, 1.2]  # SVM มีน้ำหนักมากกว่า
)

# Pipeline + Preprocessing + Ensemble
final_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', ensemble)
])

# เทรนด้วยข้อมูลที่สมดุล
final_model.fit(X_train_balanced, y_train_balanced)
""", language="python")

st.header("5. Model Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", "83.07%")
with col2:
    st.metric("Precision (Disease)", "67%")
with col3:
    st.metric("Recall (Disease)", "60%")

report = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1-Score", "Support"],
    "No Disease (0)": [0.88, 0.90, 0.89, "~1,520"],
    "Disease (1)": [0.67, 0.60, 0.63, "~483"]
})
st.dataframe(report, use_container_width=True)

st.subheader("5.1 ROC Curve & Threshold Optimization")
st.write("""
- **Default threshold (0.5)**: ทำนายแบบสมมติ (balanced)
- **Optimized threshold**: ค้นหา F1-Score ที่สูงสุดเพื่อลดผลบวกเท็จ
- มีการปรับค่า threshold เพื่อให้ Recall (จับโรคแท้) ดีขึ้น
""")

st.subheader("5.2 Feature Importance (Permutation Importance)")
st.write("**ฟีเจอร์ไหนมีผลต่อโมเดลมากที่สุด?**")
importance_example = pd.DataFrame({
    "Feature": ["humidity", "soil_pH", "temperature", "rainfall"],
    "Importance": [0.15, 0.12, 0.10, 0.08]
}).sort_values("Importance", ascending=False)
st.dataframe(importance_example, use_container_width=True)
st.write("→ **ความชื้น (humidity)** มีผลต่อการจำแนกโรคมากที่สุด")

st.header("6. Confusion Matrix & Interpretation")
st.write("""
**Confusion Matrix ของ Ensemble Model:**
```
                Predicted Negative    Predicted Positive
Actual Negative        1,375                   145
Actual Positive          194                   289
```

**ตีความผล:**
- **True Negatives (1,375)**: ทำนายถูก ไม่เป็นโรค ✓
- **False Positives (145)**: ทำนายผิด บอกเป็นโรค แต่จริงไม่เป็น ❌
- **False Negatives (194)**: ทำนายผิด บอกไม่โรค แต่จริงเป็น ❌ (ผลร้ายแรง!)
- **True Positives (289)**: ทำนายถูก เป็นโรค ✓

**การปรับปรุง:**
- False Negatives ยังสูง → ใช้ optimized threshold เพิ่ม Recall
- Soft Voting + Class Balancing ช่วยให้ประสิทธิภาพดีขึ้น
""")

st.header("7. Advantages & Limitations")
col1, col2 = st.columns(2)

with col1:
    st.success("""
    **✓ ข้อดี**
    - **เทรนเร็ว**: ไม่ต้องใช้ GPU เหมือน Deep Learning
    - **ใช้ข้อมูลตารางได้โดยตรง**: ไม่ต้อง Feature Engineering ซับซ้อน
    - **Ensemble ที่เข้มแข็ง**: รวมจุดแข็งของ 3 โมเดล
    - **Precision ดี (88%)**: ลดผลบวกเท็จ
    - **อธิบายได้ (Interpretable)**: Feature Importance ชัดเจน
    - **จัดการ Imbalance**: ใช้ Class Weights + SMOTETomek
    """)

with col2:
    st.warning("""
    **⚠ ข้อจำกัด**
    - **ไม่ใช้ข้อมูลภาพ**: ต้องแปลงเป็น Features ก่อน
    - **Recall ต่ำ (60%)**: พลาดโรคจริง ~40%
    - **ต้องเก็บสภาพแวดล้อม**: ปฏิบัติการบันทึก Temp/Humidity ต่อเนื่อง
    - **ขนาด Features น้อย**: ใช้ได้ 4 Features เท่านั้น
    - **ต้อง Preprocessing**: อาจซับซ้อนกับข้อมูลที่สกปรก
    """)

st.divider()
st.info("""
**📊 สรุป:**
- Model 1 เป็น **Baseline / Traditional ML** ที่เร็วและเข้าใจง่าย
- ใช้ได้ดีกับข้อมูลตารางอุณหภูมิ-ความชื้น-ฝน-pH
- Model 2 (CNN) ใช้ข้อมูลภาพโดยตรง → ประสิทธิภาพสูงกว่า
- **การเลือกใช้**: ขึ้นอยู่กับประเภทข้อมูลที่มี!
""")

st.divider()
st.markdown("""
### 📚 อ้างอิง Dataset
- **File**: `plant_disease_dirty.csv`
- **Kaggle**: [Plant Disease Classification Dataset](https://www.kaggle.com/datasets/turakut/plant-disease-classification)
- **Notebook**: `models/train_ml.ipynb`
""")

