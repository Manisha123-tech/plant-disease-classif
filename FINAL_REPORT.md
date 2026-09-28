# Plant Disease Classifier — Final Report

## Problem and Objective
Classify tomato and potato leaf images into five classes: Tomato healthy, Tomato Early blight, Tomato Late blight, Potato healthy, and Potato Early blight.

## Dataset
The prepared dataset contains 3,955 training images, 846 validation images, and 851 test images. The classes are imbalanced, particularly Potato healthy.

## Methods
1. HOG features and HSV color histograms with SVM and Random Forest.
2. MobileNetV2 transfer learning baseline.
3. MobileNetV2 with CBAM attention and Focal Loss.

## Results
| Model | Test accuracy |
|---|---:|
| SVM | 94.83% |
| Random Forest | 95.30% |
| MobileNetV2 baseline | 97.41% |
| MobileNetV2 + CBAM + Focal Loss | 95.65% |

The baseline achieved the highest test accuracy. The CBAM + Focal Loss variant scored 1.76 percentage points lower, so it did not improve aggregate accuracy in this experiment. This is an experimental finding, not a failure of the project.

## Application
The Streamlit demo accepts leaf images and displays a predicted class, confidence, and top three predictions. One image from each class was tested successfully. This functional check does not replace the full test-set evaluation.

## Limitations and Future Work
The dataset is imbalanced and may not represent real field conditions. Future work could include more balanced data and independent field-image testing.

## Demo Testing Results

Five sample images, one from each class, were tested using the Streamlit application.

| Actual class | Predicted class | Result |
|---|---|---|
| Tomato healthy | Tomato healthy | Correct |
| Tomato Early blight | Tomato Late blight | Incorrect |
| Tomato Late blight | Tomato Late blight | Correct |
| Potato healthy | Potato healthy | Correct |
| Potato Early blight | Potato Early blight | Correct |

Demo accuracy: 4/5 (80%).

The Tomato Early blight image was misclassified as Tomato Late blight. This small sample test is separate from the full test-set evaluation.

## Conclusion
The project implements and compares traditional machine learning and deep learning approaches and provides a working image-classification demo. All results are reported from the completed experiments.
