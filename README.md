# 🌱 Plant Disease Prediction System

ระบบทำนายโรคพืชมะเขือเทศ ด้วย Machine Learning และ Deep Learning

## 📋 โปรเจกต์นี้ประกอบด้วย

### 1️⃣ **Model 1: Ensemble Learning** (83.07% Accuracy)
- ใช้ข้อมูล: Temperature, Humidity, Rainfall, Soil pH
- อัลกอริทึม: Random Forest + SVM + XGBoost
- ประโยชน์: เร็ว, ใช้ทรัพยากรน้อย

### 2️⃣ **Model 2: Deep Learning (MobileNetV2)** (88.97% Accuracy)
- ใช้ข้อมูล: ภาพใบมะเขือเทศ (224×224 RGB)
- อัลกอริทึม: CNN + Transfer Learning
- ประโยชน์: ความแม่นยำสูง, ง่ายต่อการใช้

## 📁 โครงสร้างโปรเจกต์

```
Plant-Project/
├── app.py                          # หน้าแรก
├── pages/
│   ├── 1_Model1_Explanation.py    # อธิบาย Model 1
│   ├── 2_Model2_Explanation.py    # อธิบาย Model 2
│   ├── 3_Model1_Test.py           # ทดสอบ Model 1
│   └── 4_Model2_Test.py           # ทดสอบ Model 2
├── models/
│   ├── train_ml.ipynb             # Train Model 1
│   ├── plant_disease_mobilenetv2_fast.ipynb  # Train Model 2
│   └── model1_test_samples.csv    # ข้อมูลทดสอบ
├── data/
│   ├── plant_disease_dirty.csv    # Dataset ต้นฉบับ
│   └── Cross-validation1/         # ข้อมูลภาพทดสอบ
├── model1_final.pkl               # โมเดล Ensemble
├── model2_mobilenetv2.keras       # โมเดล CNN
└── requirements.txt               # Dependencies
```

## 🎯 โรคที่ตรวจจับได้ (10 ประเภท)

1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Target Spot
7. TYLCV
8. Tomato Mosaic Virus
9. Spider Mite
10. Healthy

## 🚀 การใช้งาน

### ติดตั้ง
```bash
pip install -r requirements.txt
```

### รันเว็บแอป
```bash
streamlit run app.py
```

## 📊 Dataset

- ขนาด: ~4,600 ภาพ train + ~4,600 ภาพ test
- ความละเอียด: 224×224 RGB
- แบ่ง: 10 ประเภท

## 📈 ผลลัพธ์

| Model | Accuracy | Speed | Size |
|-------|----------|-------|------|
| Model 1 (Ensemble) | 83.07% | เร็วมาก | 52MB |
| Model 2 (CNN) | 88.97% | ปานกลาง | 50MB |

## 🔧 เทคโนโลยี

- Streamlit, scikit-learn, TensorFlow/Keras
- XGBoost, pandas, numpy, PIL
