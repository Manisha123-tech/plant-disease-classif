# 🌿 Plant Disease Classifier

![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

An image-based deep learning application that classifies plant leaves into five categories using MobileNetV2. The project compares traditional machine learning models with deep learning architectures and provides an interactive web interface built with Streamlit.

## 📌 Project Overview

Plant diseases can significantly affect crop quality and agricultural productivity. This project uses image classification to identify selected tomato and potato leaf conditions.

The project includes:
- Dataset preparation and preprocessing
- Traditional machine learning using SVM and Random Forest
- Deep learning using MobileNetV2
- An improved MobileNetV2 architecture using CBAM and Focal Loss
- Model evaluation and comparison
- An interactive Streamlit application for image prediction

## 🎯 Classes

The classifier recognizes five categories:

| Class | Description |
|---|---|
| Tomato_healthy | Healthy tomato leaf |
| Tomato_Early_blight | Tomato early blight |
| Tomato_Late_blight | Tomato late blight |
| Potato___healthy | Healthy potato leaf |
| Potato___Early_blight | Potato early blight |

## 🧠 Models Used

### 1. Support Vector Machine (SVM)
A traditional machine learning model trained on extracted image features.

- Kernel: RBF
- C: 10
- Gamma: scale

### 2. Random Forest
An ensemble learning model trained on extracted image features.

- Number of trees: 100
- Maximum depth: 30
- Minimum samples per leaf: 2
- Class weight: Balanced

### 3. MobileNetV2 Baseline
A transfer learning model using MobileNetV2 for plant disease classification.

### 4. MobileNetV2 + CBAM + Focal Loss
An additional model that integrates:
- **CBAM:** Convolutional Block Attention Module, which helps the network focus on relevant image features.
- **Focal Loss:** A loss function designed to give more emphasis to difficult training examples.

The improved architecture is evaluated against the baseline. It does not achieve higher test accuracy in this experiment.

## 📊 Dataset

The dataset contains five classes and is divided into training, validation, and testing sets.

| Class | Training | Validation | Testing |
|---|---:|---:|---:|
| Tomato healthy | 1,113 | 238 | 240 |
| Tomato Early blight | 700 | 150 | 150 |
| Tomato Late blight | 1,336 | 286 | 287 |
| Potato healthy | 106 | 22 | 24 |
| Potato Early blight | 700 | 150 | 150 |
| **Total** | **3,955** | **846** | **851** |

## 📈 Model Performance

The models were evaluated on the test dataset.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| SVM | 94.83% | 94.78% | 94.83% | 94.76% |
| Random Forest | 95.30% | 95.27% | 95.30% | 95.27% |
| MobileNetV2 Baseline | **97.41%** | **97.43%** | **97.41%** | **97.41%** |
| MobileNetV2 + CBAM + Focal Loss | 95.65% | 95.60% | 95.65% | 95.57% |

The baseline MobileNetV2 achieved the highest test accuracy in this experiment. The CBAM and Focal Loss model provides an additional architecture for comparison, but its measured accuracy was lower.

## 🖥️ Application

The project includes a Streamlit web application that allows users to:
1. Upload a plant leaf image.
2. Run the trained model on the image.
3. View the predicted class and confidence.
4. Explore the prediction through a simple interface.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Streamlit
- OpenCV / Pillow

## 📂 Project Structure

```text
plant-disease-classifier/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── outputs/
│   ├── deep_baseline_model.keras
│   ├── improved_model.keras
│   ├── svm_best_model.pkl
│   ├── random_forest_best_model.pkl
│   ├── label_encoder.pkl
│   ├── final_comparison.json
│   ├── final_comparison.txt
│   ├── model_metrics_comparison.png
│   └── confusion matrices and training plots
│
├── 06_demo.py
├── requirements.txt
├── README.md
├── README_personA.md
├── README_personB.md
└── FINAL_REPORT.md
```

## ⚙️ Installation and Setup

### Requirements

- Python 3.13
- Git
- Windows, macOS, or Linux
- Internet connection for installing dependencies

### 1. Clone the repository

```bash
git clone https://github.com/Manisha123-tech/plant-disease-classifier.git
cd plant-disease-classifier
```

### 2. Create a virtual environment

**Windows (Command Prompt):**

```cmd
py -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Check Python version

```bash
python --version
```

Use Python 3.13 for the project environment.

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run 06_demo.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

Upload a leaf image to test the classifier.

## 📦 Trained Models

The repository includes trained model files in the `outputs/` directory. You do not need to train the models again to run the demo.

## 📉 Limitations

- The classifier recognizes only the five classes listed above.
- Predictions depend on image quality, lighting, and the appearance of the leaf.
- The dataset has fewer potato healthy samples than the other classes.
- The CBAM and Focal Loss model did not outperform the baseline in this experiment.
- The model is intended for educational purposes and should not replace professional agricultural diagnosis.

## 🚀 Future Improvements

- Add more plant species and disease categories.
- Increase the number and diversity of training images.
- Improve performance on underrepresented classes.
- Explore further model architectures and hyperparameter tuning.
- Deploy the application online for easier access.

## 👥 Project Contributors

This project was developed collaboratively, with separate contributions to traditional machine learning, deep learning, and application integration.

## 📄 License

This project is intended for educational and academic use. Add a license file if you plan to distribute it under specific reuse terms.

---

**Plant Disease Classifier** — Deep learning for plant leaf image classification.