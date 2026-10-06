# 🌱 Plant Disease Classification

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)
![AI](https://img.shields.io/badge/AI-Semester%205-green)

An academic Artificial Intelligence project for classifying tomato and potato leaf images into five classes using two syllabus-aligned AI approaches:

1. **Supervised Learning using a Feed-Forward Neural Network (MLP)**
2. **Knowledge-Based Reasoning**

The project also includes a working Streamlit application for image-based prediction.

---

## 📌 Project Overview

Plant diseases can affect crop quality and agricultural productivity.

This project explores how Artificial Intelligence can be used to classify selected tomato and potato leaf images.

The project focuses on two different AI approaches:

- Learning from labeled examples using supervised learning.
- Reasoning using class-level knowledge in a knowledge-based system.

The approaches are evaluated using a separate test dataset.

---

## 🎯 Objectives

The main objectives are:

- Analyze the plant leaf image dataset.
- Extract simple measurable image features.
- Implement a supervised classification model.
- Implement a knowledge-based reasoning approach.
- Compare the approaches quantitatively.
- Experiment with a student-designed model modification.
- Develop a working AI demonstration application.

---

## 🌿 Classes

The classifier recognizes five categories:

| Class | Description |
|---|---|
| `Tomato_healthy` | Healthy tomato leaf |
| `Tomato_Early_blight` | Tomato early blight |
| `Tomato_Late_blight` | Tomato late blight |
| `Potato___healthy` | Healthy potato leaf |
| `Potato___Early_blight` | Potato early blight |

---

# 📊 Dataset

The prepared dataset contains:

- **Training:** 3,955 images
- **Validation:** 846 images
- **Testing:** 851 images
- **Total:** 5,652 images
- **Classes:** 5

### Class Distribution

| Class | Training | Validation | Testing |
|---|---:|---:|---:|
| Tomato Healthy | 1,113 | 238 | 240 |
| Tomato Early Blight | 700 | 150 | 150 |
| Tomato Late Blight | 1,336 | 286 | 287 |
| Potato Healthy | 106 | 22 | 24 |
| Potato Early Blight | 700 | 150 | 150 |
| **Total** | **3,955** | **846** | **851** |

The dataset is imbalanced, particularly for the Potato Healthy class.

---

# 🔍 Feature Representation

The project uses eight simple image features.

Each image is resized to **128 × 128 pixels**.

The following features are extracted:

1. Mean Red
2. Mean Green
3. Mean Blue
4. Mean Brightness
5. Brightness Texture
6. Red Ratio
7. Green Ratio
8. Blue Ratio

These features are standardized using statistics calculated from the training data.

The same preprocessing is used during testing and in the Streamlit application.

---

# 🧠 Approach 1 — Supervised Learning

A feed-forward neural network (MLP) is used for supervised classification.

### Architecture

```text
8 Input Features
       ↓
Dense Layer — 32 neurons
       ↓
Dense Layer — 16 neurons
       ↓
Output Layer — 5 classes

