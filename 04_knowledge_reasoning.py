from pathlib import Path
import cv2
import numpy as np
import json

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

    green_ratio = mean_g / total if total > 0 else 0
    red_ratio = mean_r / total if total > 0 else 0

    return {
        "green_ratio": float(green_ratio),
        "red_ratio": float(red_ratio),
        "brightness": float(brightness),
        "texture": float(texture),
    }


def get_thresholds():

    values = {
        cls: []
        for cls in CLASSES
    }

    train_dir = DATASET_DIR / "train"

    for cls in CLASSES:

        class_dir = train_dir / cls

        for image_path in class_dir.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                features = extract_features(image_path)

                if features:
                    values[cls].append(features)

    thresholds = {}

    for cls in CLASSES:

        if not values[cls]:
            continue

        thresholds[cls] = {}

        for feature in values[cls][0]:

            feature_values = [
                x[feature]
                for x in values[cls]
            ]

            thresholds[cls][feature] = float(
                np.mean(feature_values)
            )

    return thresholds


def reason(features, thresholds):

    scores = {}

    for cls in CLASSES:

        if cls not in thresholds:
            continue

        score = 0

        for feature in features:

            difference = abs(
                features[feature]
                - thresholds[cls][feature]
            )

            # Smaller difference = better match
            score += difference

        scores[cls] = score

    if not scores:
        return None, {}

    prediction = min(
        scores,
        key=scores.get
    )

    return prediction, scores


def evaluate():

    print("=" * 60)
    print("KNOWLEDGE-BASED REASONING")
    print("=" * 60)

    thresholds = get_thresholds()

    correct = 0
    total = 0

    results = []

    for cls in CLASSES:

        class_dir = DATASET_DIR / "test" / cls

        for image_path in class_dir.iterdir():

            if (
                not image_path.is_file()
                or image_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            features = extract_features(image_path)

            if features is None:
                continue

            prediction, scores = reason(
                features,
                thresholds
            )

            correct += int(
                prediction == cls
            )

            total += 1

            results.append({
                "actual": cls,
                "predicted": prediction,
            })

    accuracy = correct / total

    print("\nRESULTS")
    print("-" * 40)

    print(f"Correct predictions: {correct}")
    print(f"Total predictions:   {total}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Accuracy (%):        {accuracy * 100:.2f}%")

    output = {
        "approach": "Knowledge-Based Reasoning",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "rules": [
            "Compare image features with class knowledge",
            "Select class with minimum feature difference"
        ],
    }

    output_path = (
        OUTPUT_DIR /
        "knowledge_reasoning_results.json"
    )

    with open(output_path, "w") as f:
        json.dump(
            output,
            f,
            indent=4
        )

    print(
        f"\nResults saved to:\n{output_path}"
    )


if __name__ == "__main__":
    evaluate()