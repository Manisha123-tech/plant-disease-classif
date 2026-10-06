from pathlib import Path
from collections import Counter
from PIL import Image
import json
import matplotlib.pyplot as plt

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

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


# --------------------------------------------------
# COUNT IMAGES
# --------------------------------------------------

def count_images(folder):
    return sum(
        1
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# --------------------------------------------------
# ANALYZE DATASET
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PLANT DISEASE DATASET ANALYSIS")
    print("=" * 60)

    dataset_info = {
        "classes": CLASSES,
        "splits": {}
    }

    # --------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------

    for split in SPLITS:

        split_dir = DATASET_DIR / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing dataset split: {split_dir}"
            )

        dataset_info["splits"][split] = {}

        print(f"\n{split.upper()} SET")
        print("-" * 40)

        total = 0

        for cls in CLASSES:

            class_dir = split_dir / cls

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Missing class folder: {class_dir}"
                )

            count = count_images(class_dir)

            dataset_info["splits"][split][cls] = count

            total += count

            print(f"{cls:30s}: {count}")

        dataset_info["splits"][split]["TOTAL"] = total

        print(f"{'TOTAL':30s}: {total}")

    # --------------------------------------------------
    # TOTAL DATASET
    # --------------------------------------------------

    total_images = sum(
        dataset_info["splits"][split]["TOTAL"]
        for split in SPLITS
    )

    dataset_info["total_images"] = total_images

    # --------------------------------------------------
    # IMAGE DIMENSION ANALYSIS
    # --------------------------------------------------

    print("\nIMAGE DIMENSION ANALYSIS")
    print("-" * 40)

    dimensions = Counter()

    # Analyze a maximum of 1000 images
    # to keep analysis fast.

    checked = 0
    MAX_IMAGES = 1000

    for split in SPLITS:

        split_dir = DATASET_DIR / split

        for cls in CLASSES:

            class_dir = split_dir / cls

            for image_path in class_dir.iterdir():

                if (
                    image_path.is_file()
                    and image_path.suffix.lower()
                    in IMAGE_EXTENSIONS
                ):

                    try:

                        with Image.open(image_path) as img:

                            dimensions[img.size] += 1

                            checked += 1

                    except Exception:
                        continue

                    if checked >= MAX_IMAGES:
                        break

            if checked >= MAX_IMAGES:
                break

        if checked >= MAX_IMAGES:
            break

    dataset_info["image_dimensions_sample"] = {
        f"{width}x{height}": count
        for (width, height), count in dimensions.items()
    }

    print(f"Images checked: {checked}")

    for size, count in dimensions.most_common(10):
        print(f"{size}: {count}")

    # --------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------

    json_path = OUTPUT_DIR / "dataset_analysis.json"

    with open(json_path, "w") as f:
        json.dump(dataset_info, f, indent=4)

    # --------------------------------------------------
    # CLASS DISTRIBUTION GRAPH
    # --------------------------------------------------

    train_counts = [
        dataset_info["splits"]["train"][cls]
        for cls in CLASSES
    ]

    plt.figure(figsize=(11, 6))

    plt.bar(CLASSES, train_counts)

    plt.xlabel("Plant Disease Class")
    plt.ylabel("Number of Training Images")

    plt.title("Training Dataset Class Distribution")

    plt.xticks(rotation=25, ha="right")

    plt.tight_layout()

    graph_path = OUTPUT_DIR / "dataset_class_distribution.png"

    plt.savefig(graph_path, dpi=180)

    plt.close()

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"Total images: {total_images}")
    print(f"Classes: {len(CLASSES)}")

    print(f"\nAnalysis saved to:")
    print(json_path)

    print(f"\nGraph saved to:")
    print(graph_path)


if __name__ == "__main__":
    main()