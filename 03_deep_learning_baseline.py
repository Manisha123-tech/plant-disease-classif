from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_HEAD = 8
EPOCHS_FINE = 8
NUM_CLASSES = 5

CLASS_NAMES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
]

def make_ds(split, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / split,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )
    return ds.prefetch(tf.data.AUTOTUNE)

def build_model():
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model, base

def main():
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_ds = make_ds("train", True)
    val_ds = make_ds("val", False)
    test_ds = make_ds("test", False)

    model, base = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2
        ),
    ]

    print("Stage 1: training classification head...")
    hist1 = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks
    )

    print("Stage 2: fine-tuning upper MobileNetV2 layers...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    hist2 = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    model.save(OUTPUT_DIR / "deep_baseline_model.keras")

    history = {
        "head": {k: [float(x) for x in v] for k, v in hist1.history.items()},
        "fine_tune": {k: [float(x) for x in v] for k, v in hist2.history.items()},
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }
    with open(OUTPUT_DIR / "deep_baseline_history.json", "w") as f:
        json.dump(history, f, indent=2)

    import matplotlib.pyplot as plt
    acc = history["head"]["accuracy"] + history["fine_tune"]["accuracy"]
    val_acc = history["head"]["val_accuracy"] + history["fine_tune"]["val_accuracy"]
    plt.figure(figsize=(9, 5))
    plt.plot(acc, label="Train accuracy")
    plt.plot(val_acc, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MobileNetV2 Baseline Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "deep_baseline_curve.png", dpi=180)
    plt.close()

    print(f"Baseline test accuracy: {test_acc:.4f}")
    print("Saved:", OUTPUT_DIR / "deep_baseline_model.keras")

if __name__ == "__main__":
    main()
