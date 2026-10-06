from pathlib import Path
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
]

CLASS_TO_INDEX = {
    cls: index
    for index, cls in enumerate(CLASSES)
}

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):
    """
    Extract the same simple numerical features
    used in 02_feature_analysis.py.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.resize(image, (128, 128))

    # OpenCV loads images as BGR
    b, g, r = cv2.split(image)

    # Mean RGB values
    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness statistics
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    # Relative color ratios
    total = mean_r + mean_g + mean_b

    if total > 0:
        red_ratio = mean_r / total
        green_ratio = mean_g / total
        blue_ratio = mean_b / total
    else:
        red_ratio = 0.0
        green_ratio = 0.0
        blue_ratio = 0.0

    return [
        mean_r,
        mean_g,
        mean_b,
        mean_brightness,
        std_brightness,
        red_ratio,
        green_ratio,
        blue_ratio,
    ]


# ============================================================
# LOAD DATASET
# ============================================================

def load_split(split):

    X = []
    y = []

    split_dir = DATASET_DIR / split

    print(f"\nLoading {split} dataset...")

    for cls in CLASSES:

        class_dir = split_dir / cls

        image_files = [
            p
            for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_files:

            features = extract_features(image_path)

            if features is not None:

                X.append(features)
                y.append(CLASS_TO_INDEX[cls])

        print(
            f"{cls:30s}: "
            f"{len(image_files)} images"
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SUPERVISED LEARNING - BASELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD TRAIN / VALIDATION / TEST
    # --------------------------------------------------------

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print("\nDataset shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val  :", X_val.shape)
    print("y_val  :", y_val.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    # --------------------------------------------------------
    # NORMALIZE FEATURES
    # --------------------------------------------------------

    # Calculate statistics ONLY from training data.
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    # Prevent division by zero.
    train_std[train_std == 0] = 1.0

    X_train_scaled = (X_train - train_mean) / train_std
    X_val_scaled = (X_val - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std

    # --------------------------------------------------------
    # BUILD NEURAL NETWORK
    # --------------------------------------------------------

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            len(CLASSES),
            activation="softmax"
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nModel:")
    model.summary()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    history = model.fit(
        X_train_scaled,
        y_train,

        validation_data=(
            X_val_scaled,
            y_val
        ),

        epochs=30,
        batch_size=32,

        verbose=1,

        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ],
    )

    # --------------------------------------------------------
    # TEST PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test_scaled,
        verbose=0
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASSES,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(9, 7))

    plt.imshow(cm)

    plt.title(
        "Supervised Learning - Confusion Matrix"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        range(len(CLASSES)),
        CLASSES,
        rotation=35,
        ha="right"
    )

    plt.yticks(
        range(len(CLASSES)),
        CLASSES
    )

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    cm_path = (
        OUTPUT_DIR /
        "supervised_learning_confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # ACCURACY CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    accuracy_curve_path = (
        OUTPUT_DIR /
        "supervised_learning_accuracy_curve.png"
    )

    plt.savefig(
        accuracy_curve_path,
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # LOSS CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    loss_curve_path = (
        OUTPUT_DIR /
        "supervised_learning_loss_curve.png"
    )

    plt.savefig(
        loss_curve_path,
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        OUTPUT_DIR /
        "supervised_learning_model.keras"
    )

    model.save(model_path)

    # --------------------------------------------------------
    # SAVE SCALING INFORMATION
    # --------------------------------------------------------

    scaler_path = (
        OUTPUT_DIR /
        "supervised_learning_scaler.npz"
    )

    np.savez(
        scaler_path,
        mean=train_mean,
        std=train_std,
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = {
        "approach": "Supervised Learning",
        "model": "Feed-Forward Neural Network",
        "features": [
            "mean_r",
            "mean_g",
            "mean_b",
            "mean_brightness",
            "std_brightness",
            "red_ratio",
            "green_ratio",
            "blue_ratio",
        ],
        "classes": CLASSES,
        "train_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "test_images": int(len(y_test)),
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }

    results_path = (
        OUTPUT_DIR /
        "supervised_learning_results.json"
    )

    with open(results_path, "w") as f:
        json.dump(
            results,
            f,
            indent=4
        )

    print("\n" + "=" * 60)
    print("SUPERVISED LEARNING COMPLETE")
    print("=" * 60)

    print(
        f"Model saved to:\n{model_path}"
    )

    print(
        f"\nResults saved to:\n{results_path}"
    )

    print(
        f"\nConfusion matrix:\n{cm_path}"
    )

    print(
        f"\nAccuracy curve:\n{accuracy_curve_path}"
    )

    print(
        f"\nLoss curve:\n{loss_curve_path}"
    )


if __name__ == "__main__":
    main()