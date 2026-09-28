
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Import the custom classes used by the improved model.
from model_components import CBAM, FocalLoss

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"

CLASS_NAMES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


def load_test_dataset():
    return tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "test",
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def evaluate_deep_model(model_path, name, test_ds, y_true):
    print(f"\nEvaluating {name}...")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "CBAM": CBAM,
            "FocalLoss": FocalLoss,
        },
        compile=False,
    )

    probabilities = model.predict(test_ds, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    return {
        "model": name,
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classification_report": report,
        "predictions": predictions,
    }


def save_confusion_matrix(name, y_true, predictions):
    cm = confusion_matrix(
        y_true,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
    )
    ax.set_title(f"{name} Confusion Matrix")
    fig.tight_layout()

    filename = name.lower().replace(" ", "_").replace("+", "plus")
    fig.savefig(
        OUTPUT_DIR / f"{filename}_confusion_matrix.png",
        dpi=160,
    )
    plt.close(fig)


def save_comparison_chart(rows):
    metrics = [
        ("accuracy", "Accuracy"),
        ("precision_weighted", "Weighted Precision"),
        ("recall_weighted", "Weighted Recall"),
        ("f1_weighted", "Weighted F1"),
    ]

    names = [row["model"] for row in rows]
    x = np.arange(len(names))
    width = 0.19

    fig, ax = plt.subplots(figsize=(13, 7))

    for i, (key, label) in enumerate(metrics):
        values = [row[key] for row in rows]
        ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=label,
        )

    ax.set_ylabel("Score")
    ax.set_title("Plant Disease Classification Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "model_metrics_comparison.png",
        dpi=180,
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test dataset...")
    test_ds = load_test_dataset()

    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    rows = []

    # Load measured SVM and Random Forest results.
    shallow_path = OUTPUT_DIR / "shallow_results.json"

    if shallow_path.exists():
        print("\nLoading shallow-learning results...")
        data = json.loads(
            shallow_path.read_text(encoding="utf-8")
        )

        for result in data.get("results", []):
            rows.append({
                "model": result["model"],
                "accuracy": result["accuracy"],
                "precision_weighted": result["precision_weighted"],
                "recall_weighted": result["recall_weighted"],
                "f1_weighted": result["f1_weighted"],
                "classification_report": result.get(
                    "classification_report"
                ),
            })
    else:
        print("Warning: shallow_results.json not found.")

    deep_models = [
        (
            OUTPUT_DIR / "deep_baseline_model.keras",
            "MobileNetV2 Baseline",
        ),
        (
            OUTPUT_DIR / "improved_model.keras",
            "MobileNetV2 + CBAM + Focal Loss",
        ),
    ]

    for model_path, name in deep_models:
        if not model_path.exists():
            print(f"Warning: model not found: {model_path}")
            continue

        result = evaluate_deep_model(
            model_path,
            name,
            test_ds,
            y_true,
        )

        save_confusion_matrix(
            name,
            y_true,
            result["predictions"],
        )
        result.pop("predictions")
        rows.append(result)

    if not rows:
        raise FileNotFoundError("No model results were found.")

    # Save full comparison data.
    with open(
        OUTPUT_DIR / "final_comparison.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(rows, f, indent=2)

    # Save a readable report.
    with open(
        OUTPUT_DIR / "final_comparison.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("PLANT DISEASE CLASSIFIER: MODEL COMPARISON\n")
        f.write("=" * 70 + "\n")
        f.write(f"Test images: {len(y_true)}\n\n")

        for result in rows:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(
                f"Weighted precision: "
                f"{result['precision_weighted']:.4f}\n"
            )
            f.write(
                f"Weighted recall: "
                f"{result['recall_weighted']:.4f}\n"
            )
            f.write(
                f"Weighted F1: "
                f"{result['f1_weighted']:.4f}\n"
            )
            f.write("-" * 70 + "\n")

    save_comparison_chart(rows)

    print("\nFINAL MODEL COMPARISON")
    print("=" * 65)

    for result in rows:
        print(f"\n{result['model']}")
        print(f"Accuracy:  {result['accuracy']:.4%}")
        print(f"Precision: {result['precision_weighted']:.4%}")
        print(f"Recall:    {result['recall_weighted']:.4%}")
        print(f"F1-score:  {result['f1_weighted']:.4%}")

    print("\nComparison completed successfully.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
