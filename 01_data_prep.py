
from pathlib import Path
import shutil
import random
import json
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = Path.home() / "Downloads" / "29Sept2026" / "archive" / "PlantVillage"
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"

CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
]

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check that all five source folders exist.
    for cls in CLASSES:
        if not (SOURCE_DIR / cls).is_dir():
            raise FileNotFoundError(f"Missing class folder: {cls}")

    # Collect images and check for empty folders.
    images_by_class = {}
    for cls in CLASSES:
        images = [
            p for p in (SOURCE_DIR / cls).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            raise RuntimeError(f"No images found in {cls}")
        random.shuffle(images)
        images_by_class[cls] = images

    # Create clean output split folders.
    for split in SPLITS:
        split_dir = DATASET_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for cls in CLASSES:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)

    counts = {}
    for cls, images in images_by_class.items():
        n = len(images)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])

        split_images = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        counts[cls] = {}
        for split, files in split_images.items():
            destination = DATASET_DIR / split / cls
            for image_path in files:
                shutil.copy2(image_path, destination / image_path.name)
            counts[cls][split] = len(files)

        print(f"{cls}: {counts[cls]}")

    with open(OUTPUT_DIR / "dataset_info.json", "w") as f:
        json.dump(
            {"classes": CLASSES, "counts": counts, "seed": SEED},
            f, indent=2
        )

    train_counts = [counts[c]["train"] for c in CLASSES]
    plt.figure(figsize=(11, 6))
    plt.bar(CLASSES, train_counts)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Number of training images")
    plt.title("Training Class Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=180)
    plt.close()

    print("\nDataset preparation complete!")
    print(f"Dataset saved to: {DATASET_DIR}")
    print(f"Summary saved to: {OUTPUT_DIR / 'dataset_info.json'}")

if __name__ == "__main__":
    main()
