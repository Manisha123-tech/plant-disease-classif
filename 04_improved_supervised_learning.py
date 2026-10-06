from pathlib import Path
import cv2
import numpy as np
import json
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"

CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_features(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.resize(image, (128, 128))

    b, g, r = cv2.split(image)

    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    texture = np.std(gray)

    total = mean_r + mean_g + mean_b

    red_ratio = mean_r / total if total > 0 else 0
    green_ratio = mean_g / total if total > 0 else 0
    blue_ratio = mean_b / total if total > 0 else 0

    return [
        mean_r,
        mean_g,
        mean_b,
        brightness,
        texture,
        red_ratio,
        green_ratio,
        blue_ratio
    ]


def load_split(split):

    X = []
    y = []

    split_dir = DATASET_DIR / split

    for class_index, class_name in enumerate(CLASSES):

        class_dir = split_dir / class_name

        for image_path in class_dir.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                features = extract_features(image_path)

                if features is not None:
                    X.append(features)
                    y.append(class_index)

    return np.array(X, dtype=np.float32), np.array(y)


print("=" * 60)
print("IMPROVED SUPERVISED LEARNING")
print("=" * 60)

print("\nLoading dataset...")

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")

# Standardization using TRAINING data only
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std == 0] = 1

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# ---------------------------------------------------------
# IMPROVED MODEL
# ---------------------------------------------------------

model = Sequential([
    Input(shape=(8,)),

    Dense(64, activation="relu"),
    Dense(32, activation="relu"),

    Dense(5, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


print("\nTraining improved model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

print("\nEvaluating on test set...")

probabilities = model.predict(
    X_test,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("IMPROVED MODEL RESULTS")
print("=" * 60)

print(f"Accuracy :  {accuracy * 100:.2f}%")
print(f"Precision:  {precision * 100:.2f}%")
print(f"Recall   :  {recall * 100:.2f}%")
print(f"F1 Score :  {f1 * 100:.2f}%")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=CLASSES,
        zero_division=0
    )
)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

results = {
    "approach": "Improved Supervised Learning",
    "architecture": [
        "Input: 8 features",
        "Dense: 64 neurons",
        "Dense: 32 neurons",
        "Output: 5 classes"
    ],
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "epochs_trained": len(history.history["loss"])
}

with open(
    OUTPUT_DIR / "improved_supervised_results.json",
    "w"
) as f:
    json.dump(results, f, indent=4)


np.save(
    OUTPUT_DIR / "improved_scaler_mean.npy",
    mean
)

np.save(
    OUTPUT_DIR / "improved_scaler_std.npy",
    std
)

model.save(
    OUTPUT_DIR / "improved_supervised_model.keras"
)

np.save(
    OUTPUT_DIR / "improved_predictions.npy",
    predictions
)

np.save(
    OUTPUT_DIR / "improved_test_labels.npy",
    y_test
)

print("\nResults saved successfully.")