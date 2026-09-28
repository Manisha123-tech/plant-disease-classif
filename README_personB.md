# Person B — Deep Learning, Improvement, and Demo

## Baseline
Implemented transfer learning using ImageNet-pretrained MobileNetV2 with a five-class classification head. The measured test accuracy was 97.41%.

## Student-Designed Project Improvement
Integrated a CBAM attention block and Focal Loss into a second MobileNetV2 model. CBAM applies channel and spatial attention; Focal Loss emphasizes harder examples during training. These are established methods; the project contribution is their integration and evaluation for this task.

The improved model achieved 95.65% test accuracy, 1.76 percentage points below the baseline. Therefore, the change did not improve aggregate accuracy in this run.

## Demo
Run `streamlit run 06_demo.py` from the project directory. Upload a JPG, JPEG, or PNG leaf image and click Predict disease. The app displays the predicted class, confidence, and top three predictions. One image from each of the five classes was tested successfully.

## Outputs
See `outputs/deep_baseline_model.keras`, `outputs/deep_baseline_curve.png`, `outputs/improved_model.keras`, `outputs/improved_model_curve.png`, `outputs/model_metrics_comparison.png`, and the confusion matrices.

## Limitations
The dataset is imbalanced and may not represent real-world field conditions. Confidence is not a guarantee of correctness.
