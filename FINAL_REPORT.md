# Plant Disease Classification — Final Report

## 1. Problem and Objective

The objective of this project is to classify plant leaf images into five classes:

1. Tomato Healthy
2. Tomato Early Blight
3. Tomato Late Blight
4. Potato Healthy
5. Potato Early Blight

The project implements two AI approaches aligned with the Artificial Intelligence syllabus:

1. Supervised Learning
2. Knowledge-Based Reasoning

A working Streamlit application was also developed for image-based prediction.

---

## 2. Dataset

The prepared dataset contains:

- Training images: 3,955
- Validation images: 846
- Test images: 851
- Total images: 5,652
- Number of classes: 5

The dataset is imbalanced. In particular, the Potato Healthy class contains substantially fewer images than some of the other classes.

### Class Distribution

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Tomato Healthy | 1,113 | 238 | 240 |
| Tomato Early Blight | 700 | 150 | 150 |
| Tomato Late Blight | 1,336 | 286 | 287 |
| Potato Healthy | 106 | 22 | 24 |
| Potato Early Blight | 700 | 150 | 150 |
| **Total** | **3,955** | **846** | **851** |

---

## 3. Feature Representation

Instead of directly using the complete image as input, eight simple measurable image features were extracted.

The features are:

- Mean Red value
- Mean Green value
- Mean Blue value
- Mean brightness
- Brightness texture (standard deviation)
- Red ratio
- Green ratio
- Blue ratio

Images are resized to 128 × 128 pixels before feature extraction.

The feature values are standardized using statistics calculated from the training set.

The same preprocessing is applied to the validation set, test set, and demo images.

---

# 4. AI Approach 1 — Supervised Learning

A feed-forward neural network (MLP) was implemented as the supervised learning approach.

### Architecture

```text
8 input features
       ↓
Dense Layer — 32 neurons
       ↓
Dense Layer — 16 neurons
       ↓
Output Layer — 5 classes