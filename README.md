# Plant Disease Classifier

A reproducible academic project comparing classical machine learning and deep learning for five plant-leaf classes.

## Classes
- Tomato___healthy
- Tomato___Early_blight
- Tomato___Late_blight
- Potato___healthy
- Potato___Early_blight

## Models
1. SVM using HOG + HSV color histogram
2. Random Forest using HOG + HSV color histogram
3. MobileNetV2 transfer-learning baseline
4. Improved MobileNetV2 with CBAM attention + Focal Loss

## Workflow

### 1. Prepare the dataset
```bash
python 01_data_prep.py
```

### 2. Train classical ML
```bash
python 02_shallow_learning.py
```

### 3. Train MobileNetV2 baseline
```bash
python 03_deep_learning_baseline.py
```

### 4. Train improved model
```bash
python 04_improved_model.py
```

### 5. Build final comparison
```bash
python 05_compare_results.py
```

### 6. Run demo
```bash
streamlit run 06_demo.py
```

The scripts create their own output files. Do not add fake metrics; use the metrics produced by the experiments.
