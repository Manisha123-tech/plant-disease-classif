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

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_features(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.resize(image, (128, 128))

    b, g, r = cv2.split(image)

    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    total = mean_r + mean_g + mean_b

    if total > 0:
        red_ratio = mean_r / total
        green_ratio = mean_g / total
        blue_ratio = mean_b / total
    else:
        red_ratio = 0.0
        green_ratio = 0.0
        blue_ratio = 0.0

    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "red_ratio": red_ratio,
        "green_ratio": green_ratio,
        "blue_ratio": blue_ratio,
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)

    summary = {}

    for split in SPLITS:

        print(f"\nProcessing {split}...")

        summary[split] = {}

        for cls in CLASSES:

            class_dir = DATASET_DIR / split / cls

            image_files = [
                p for p in class_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            ]

            feature_values = []

            for image_path in image_files:

                features = extract_features(image_path)

                if features is not None:
                    feature_values.append(features)

            print(
                f"{cls:30s}: "
                f"{len(feature_values)} images processed"
            )

            if not feature_values:
                continue

            summary[split][cls] = {}

            feature_names = feature_values[0].keys()

            for feature in feature_names:

                values = [
                    item[feature]
                    for item in feature_values
                ]

                summary[split][cls][feature] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }

    output_file = OUTPUT_DIR / "feature_analysis.json"

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"Saved to:")
    print(output_file)


if __name__ == "__main__":
    main()