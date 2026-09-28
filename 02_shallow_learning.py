
from pathlib import Path
import json
import time

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_SIZE = (128, 128)

CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
]


def extract_features(path):
    img = cv2.imread(str(path))

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    img = cv2.resize(img, IMAGE_SIZE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HOG features capture edges and texture patterns.
    hog_features = hog(
        rgb,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        channel_axis=-1,
        block_norm="L2-Hys",
    )

    # Color histograms capture the distribution of colors.
    hist_features = []

    for channel, bins, rng in [
        (0, 32, (0, 180)),
        (1, 32, (0, 256)),
        (2, 32, (0, 256)),
    ]:
        hist = cv2.calcHist(
            [hsv], [channel], None, [bins], list(rng)
        ).flatten()

        hist = hist / (hist.sum() + 1e-8)
        hist_features.extend(hist)

    return np.concatenate([
        hog_features,
        np.asarray(hist_features, dtype=np.float32),
    ])


def load_split(split):
    X, y = [], []
    split_dir = DATASET_DIR / split

    for cls in CLASSES:
        class_dir = split_dir / cls

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        image_paths = []
        for extension in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            image_paths.extend(class_dir.glob(extension))

        image_paths = sorted(set(image_paths))
        print(f"{split}: {cls}: {len(image_paths)} images")

        for path in image_paths:
            X.append(extract_features(path))
            y.append(cls)

    return np.asarray(X, dtype=np.float32), np.asarray(y)


def evaluate(name, model, X, y, label_encoder):
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
    )

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        OUTPUT_DIR / f"{filename}_confusion_matrix.png",
        dpi=180,
    )
    plt.close()

    return {
        "model": name,
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classification_report": report,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print("Extracting training features...")
    X_train, y_train = load_split("train")

    print("Extracting validation features...")
    X_val, y_val = load_split("val")

    print("Extracting test features...")
    X_test, y_test = load_split("test")

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)

    y_train_i = label_encoder.transform(y_train)
    y_val_i = label_encoder.transform(y_val)
    y_test_i = label_encoder.transform(y_test)

    # Combine training and validation data for final model fitting.
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train_i, y_val_i])

    print("\nTraining SVM...")

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=10,
            gamma="scale",
            kernel="rbf",
            probability=True,
            random_state=42,
        )),
    ])

    svm.fit(X_combined, y_combined)
    best_svm = svm

    print("\nTraining Random Forest...")

    best_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=30,
        min_samples_leaf=2,
    )

    best_rf.fit(X_combined, y_combined)

    print("\nEvaluating SVM...")
    svm_result = evaluate(
        "SVM", best_svm, X_test, y_test_i, label_encoder
    )

    print("\nEvaluating Random Forest...")
    rf_result = evaluate(
        "Random Forest", best_rf, X_test, y_test_i, label_encoder
    )

    # Save trained models and label encoder.
    joblib.dump(best_svm, OUTPUT_DIR / "svm_best_model.pkl")
    joblib.dump(best_rf, OUTPUT_DIR / "random_forest_best_model.pkl")
    joblib.dump(label_encoder, OUTPUT_DIR / "label_encoder.pkl")

    results = {
        "svm_parameters": {
            "C": 10,
            "gamma": "scale",
            "kernel": "rbf",
        },
        "random_forest_parameters": {
            "n_estimators": 100,
            "max_depth": 30,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
        "results": [svm_result, rf_result],
        "runtime_seconds": time.time() - start,
    }

    with open(
        OUTPUT_DIR / "shallow_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2)

    with open(
        OUTPUT_DIR / "shallow_results.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for result in results["results"]:
            f.write(f"{result['model']}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision_weighted']:.4f}\n")
            f.write(f"Recall: {result['recall_weighted']:.4f}\n")
            f.write(f"F1-score: {result['f1_weighted']:.4f}\n\n")
            f.write(result["classification_report"])
            f.write("\n\n")

    print("\nShallow-learning training complete.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
