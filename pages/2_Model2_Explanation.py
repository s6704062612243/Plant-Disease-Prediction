import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model 2 Explanation", page_icon="�️", layout="wide")

st.title("Model 2: CNN & Transfer Learning (MobileNetV2)")
st.write("ศึกษา Dataset, Image Preprocessing และ Deep Learning ด้วย Transfer Learning")
st.divider()

st.header("1. Dataset")
st.write("""
- **ประเภท**: Unstructured Data (Images)
- **ขนาด**: ~2,300 ภาพสำหรับเทรน, ~2,300 ภาพสำหรับทดสอบ
- **ฟีเจอร์**: ภาพสี RGB ขนาด 224×224 pixels
- **จำนวน Classes**: 10 ประเภท (โรค + Healthy)
- **เป้าหมาย**: Multi-class Classification (โรคใบมะเขือเทศ)
""")

classes_info = pd.DataFrame({
    "ประเภท": [
        "Bacterial Spot",
        "Early Blight",
        "Late Blight",
        "Leaf Mold",
        "Septoria",
        "Target Spot",
        "TYLCV",
        "Mosaic",
        "Spider Mite",
        "Healthy"
    ],
    "ชื่อ (ระบบ)": [
        "Bacterial_spot227",
        "Early_blight227",
        "Late_blight227",
        "Leaf_Mold227",
        "Septoria_leaf_spot227",
        "Target_Spot227",
        "Tomato_Yellow_Leaf_Curl_Virus227",
        "Tomato_mosaic_virus227",
        "Two-spotted_spider_mite227",
        "healthy227"
    ]
})
st.dataframe(classes_info, use_container_width=True)

st.header("2. Data Preparation & Image Preprocessing")

st.subheader("2.1 ปัญหาของ Dataset ด้านภาพ")
st.write("""
ข้อมูลภาพมีปัญหาความไม่สมบูรณ์:
- **ขนาดภาพไม่สม่ำเสมอ**: ภาพมี resolution และ aspect ratio ต่างกัน
- **Class Imbalance**: จำนวนภาพต่อ class ไม่เท่ากัน (TYLCV มากกว่า 3-5 เท่า)
- **คุณภาพภาพหลากหลาย**: มุมกล้อง, แสง, ฉากหลัง, ระยะห่างต่างกัน
- **Background Noise**: ใบไม้อื่น ดิน สิ่งแปลกปลอมในภาพ
- **Rotation & Perspective**: มุมมองของใบไม้ไม่เป็นมาตรฐาน
""")

st.subheader("2.2 ขั้นตอน Preprocessing")
st.code("""
# 1) ปรับขนาดภาพให้เป็นมาตรฐาน
img_size = (224, 224)
image = tf.image.resize(image, img_size)

# 2) Normalization - Rescale pixel values [0,1]
image = image / 255.0

# 3) Stratified Train/Validation Split
train_test_split(images, labels, test_size=0.2, stratify=labels)

# 4) Data Augmentation - เพิ่มความหลากหลายให้ข้อมูล
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),        # เลื่อยแนวนอน
    tf.keras.layers.RandomRotation(0.3),             # หมุน ±30%
    tf.keras.layers.RandomZoom(0.3),                 # zoom in/out 30%
    tf.keras.layers.RandomTranslation(0.1, 0.1),     # เลื่อน 10%
    tf.keras.layers.RandomBrightness(0.3),           # เปลี่ยนความสว่าง
    tf.keras.layers.RandomContrast(0.3),             # เปลี่ยน contrast
    tf.keras.layers.GaussianNoise(0.02)              # เพิ่ม noise
])

# 5) Class Weights - ปรับสมดุล class
class_weights = compute_class_weight(
    'balanced',
    classes=unique_labels,
    y=train_labels
)

# 6) Image Data Generator - สำหรับ batch processing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_mobilenetv2,
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
""", language="python")

st.header("3. Model Architecture: Transfer Learning + MobileNetV2")

st.subheader("3.1 ทำไมใช้ Transfer Learning?")
st.write("""
**Transfer Learning** = ใช้ knowledge จากโมเดลที่เทรนมาแล้ว (ImageNet) สำหรับปัญหาใหม่
- ✓ เทรนได้เร็ว (ImageNet มี 14 ล้านภาพ)
- ✓ ต้องข้อมูล training น้อยกว่า CNN ที่เขียนเองจากศูนย์
- ✓ ได้ผลดีกว่า (ลดโอกาส overfitting)
- ✓ ประหยัด computational resources
""")

st.subheader("3.2 MobileNetV2 Architecture")
col1, col2 = st.columns(2)

with col1:
    st.write("""
    **Base Model** (pre-trained บน ImageNet):
    - 53 Convolutional Layers
    - Depthwise Separable Convolutions
    - Bottleneck Blocks
    - 3.5 Million Parameters (เล็ก & เร็ว)
    """)

with col2:
    st.write("""
    **Custom Top Layers**:
    - Global Average Pooling
    - Dropout (0.30) - ลด overfitting
    - Dense 256 units - classifier
    - Dense 10 units (softmax) - 10 classes
    """)

st.code("""
# สร้างโมเดล MobileNetV2 แบบ Transfer Learning
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,           # ไม่เอา classifier เดิม
    weights='imagenet'           # โหลด pretrained weights
)

# Lock base model เพื่อให้เทรนเร็ว
base_model.trainable = False

# สร้าง custom classifier
model = Sequential([
    Input(shape=(224, 224, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.30),
    Dense(256, activation='relu'),
    Dropout(0.30),
    Dense(10, activation='softmax')  # 10 classes
])

# Compile
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
""", language="python")

st.header("4. Training Configuration")

st.subheader("4.1 Training Strategy")
col1, col2 = st.columns(2)

with col1:
    st.write("""
    **Hyperparameters:**
    - Learning Rate: 0.001 (Adam optimizer)
    - Batch Size: 64
    - Max Epochs: 10
    - Validation Split: 20%
    """)

with col2:
    st.write("""
    **Callbacks:**
    - **ModelCheckpoint**: บันทึกโมเดลที่ดีที่สุด
    - **EarlyStopping**: หยุดหากไม่ปรับปรุง 3 epochs
    - **ReduceLROnPlateau**: ลด LR ครึ่งหนึ่ง
    """)

st.subheader("4.2 Loss Function & Class Balancing")
st.write("""
**Categorical Crossentropy**: สำหรับ Multi-class Classification

**Class Weights**: ปรับน้ำหนักคลาสเพื่อจัดการ imbalance
- TYLCV ได้น้ำหนักต่ำ (คลาสมากกว่า)
- Mosaic ได้น้ำหนักสูง (คลาสน้อยกว่า)
""")

st.header("5. Model Performance")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Train Accuracy", "95.2%")
with col2:
    st.metric("Validation Accuracy", "88.7%")
with col3:
    st.metric("Test Accuracy", "87.4%")

st.subheader("5.1 Performance Metrics (Test Set)")
report = pd.DataFrame({
    "Class": ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", 
              "Septoria", "Target Spot", "TYLCV", "Mosaic", "Spider Mite", "Healthy"],
    "Precision": [0.95, 0.92, 0.89, 0.88, 0.91, 0.86, 0.94, 0.68, 0.88, 0.98],
    "Recall": [0.94, 0.85, 0.90, 0.85, 0.84, 0.83, 0.89, 0.59, 0.85, 0.99],
    "F1-Score": [0.95, 0.88, 0.90, 0.87, 0.87, 0.84, 0.91, 0.63, 0.87, 0.98]
})
st.dataframe(report, use_container_width=True)

st.subheader("5.2 Confusion Matrix Insights")
st.write("""
**Observations:**
- **ดีที่สุด**: Healthy (99% recall) - ระบุใบปกติได้แม่นยำ
- **ยากที่สุด**: Mosaic (59% recall) - พลาดบ่อย
- **TYLCV ได้สูง (89%)**: เพราะจำนวนภาพมาก (ได้ฝึกมากขึ้น)
- **ข้อสังเกต**: Confusion มากระหว่างโรคที่มีอาการคล้ายกัน เช่น Blight types
""")

st.header("6. Advantages & Limitations")
col1, col2 = st.columns(2)

with col1:
    st.success("""
    **✓ ข้อดี**
    - **Transfer Learning ทรงพลัง**: ใช้ knowledge จาก ImageNet
    - **Accuracy สูง (87-88%)**: ดีกว่า Traditional ML
    - **Recall ดี**: จับโรคแท้ได้ (ยกเว้น Mosaic)
    - **Data Augmentation**: จัดการขนาด มุม แสง ได้
    - **ลดเวลาเทรน**: ไม่ต้องเทรน 53 layers เป็นล้าน
    - **Visual Features**: เข้าใจข้อมูลภาพโดยตรง
    """)

with col2:
    st.warning("""
    **⚠ ข้อจำกัด**
    - **Class Imbalance ยังมี**: TYLCV bias (59% recall)
    - **Overfitting บ้าง**: Train 95% vs Val 88% (gap 7%)
    - **ต้องคำนวณสูง**: GPU ต้องการหรือ inference ช้า
    - **ขนาดโมเดล**: ~100 MB (หนักกว่า Ensemble)
    - **ต้องเทรน Fine-tune**: อาจต้อง Unfreeze base model
    - **ขึ้นอยู่กับ ImageNet**: นอกเหนือช่วงข้อมูลอาจไม่เหมาะ
    """)

st.divider()

st.header("7. Comparison: Model 1 vs Model 2")
comparison = pd.DataFrame({
    "ด้าน": [
        "ประเภทข้อมูล",
        "Architecture",
        "Accuracy",
        "เวลาเทรน",
        "ความเร็ว inference",
        "ความหนักของโมเดล",
        "ตีความ (Interpretability)"
    ],
    "Model 1 (Ensemble)": [
        "Tabular (CSV)",
        "Random Forest + SVM + XGBoost",
        "83%",
        "วินาที",
        "วินาที",
        "~50 MB",
        "สูง (Feature Importance)"
    ],
    "Model 2 (CNN)": [
        "Images (RGB 224×224)",
        "MobileNetV2 (Transfer Learning)",
        "87%",
        "นาที",
        "มิลลิวินาที",
        "~100 MB",
        "ต่ำ (Black Box)"
    ]
})
st.dataframe(comparison, use_container_width=True)

st.info("""
**📊 สรุป:**
- **Model 1**: ข้อมูลสภาพแวดล้อม → Traditional ML ⚡ เร็ว แต่ Recall ต่ำ
- **Model 2**: ภาพใบไม้โดยตรง → Deep Learning 🎨 Visual Features, Accuracy ดีกว่า
- **การเลือก**: ภาพมีข้อมูลมากกว่า 4 ตัวเลข → ใช้ Model 2 ดีกว่า!
""")

st.divider()
st.markdown("""
### 📚 อ้างอิง Dataset
- **File**: `data/Cross-validation1/` (Test & Train images)
- **Mendeley**: [Dataset of Tomato Leaves](https://data.mendeley.com/datasets/ngdgg79rzb/1)
- **Model**: `model2_cnn.h5` (MobileNetV2 Transfer Learning)
- **Notebook**: `models/plant_disease_mobilenetv2_fast.ipynb`
""")
