# Person A — Shallow Learning

## Work
Prepared the dataset and implemented HOG and HSV color-histogram feature extraction. Trained an RBF-kernel SVM and a Random Forest classifier.

## Results
| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
|---|---:|---:|---:|---:|
| SVM | 94.83% | 94.78% | 94.83% | 94.76% |
| Random Forest | 95.30% | 95.27% | 95.30% | 95.27% |

Random Forest scored 0.47 percentage points higher than SVM in this experiment. HOG represents edge and shape patterns; HSV histograms represent color distribution. These engineered features provide a traditional machine-learning comparison to deep learning.

## Outputs
See `outputs/shallow_results.txt`, `outputs/shallow_results.json`, `outputs/svm_best_model.pkl`, `outputs/random_forest_best_model.pkl`, `outputs/label_encoder.pkl`, and the confusion-matrix images.

## Note
The models were trained with selected fixed parameters; a formal grid search was not performed. The dataset is imbalanced, so per-class performance should also be considered.
