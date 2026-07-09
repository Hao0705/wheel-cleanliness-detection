import os
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

DATASET_DIR = "../dataset"

IMAGE_DIR = Path(DATASET_DIR) / "images"
LABEL_DIR = Path(DATASET_DIR) / "labels"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Số class của dataset
NUM_CLASSES = 2

# =====================================================
# LOAD IMAGE FILES
# =====================================================

image_files = []

for ext in IMAGE_EXTENSIONS:
    image_files.extend(IMAGE_DIR.glob(f"*{ext}"))
    image_files.extend(IMAGE_DIR.glob(f"*{ext.upper()}"))

image_dict = {img.stem: img for img in image_files}

label_files = list(LABEL_DIR.glob("*.txt"))
label_dict = {lb.stem: lb for lb in label_files}

print("=" * 60)
print("DATASET CHECK")
print("=" * 60)

# =====================================================
# ERROR CONTAINERS
# =====================================================

missing_labels = []
missing_images = []
empty_labels = []
invalid_columns = []
invalid_class = []

# =====================================================
# 1. Missing Label
# =====================================================

for stem in image_dict:
    if stem not in label_dict:
        missing_labels.append(stem)

# =====================================================
# 2. Missing Image
# =====================================================

for stem in label_dict:
    if stem not in image_dict:
        missing_images.append(stem)

# =====================================================
# 3. Filename Check
# =====================================================
# Nếu thiếu image hoặc label thì coi như tên không khớp

filename_errors = sorted(set(missing_labels + missing_images))

# =====================================================
# 4,5,6 Check Label File
# =====================================================

for stem, label_path in label_dict.items():

    if not label_path.exists():
        continue

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    # ---------- Empty label ----------
    if len(lines) == 0:
        empty_labels.append(label_path.name)
        continue

    for line_number, line in enumerate(lines, start=1):

        if line == "":
            continue

        parts = line.split()

        # ---------- Wrong columns ----------
        if len(parts) != 5:
            invalid_columns.append(
                (
                    label_path.name,
                    line_number,
                    line
                )
            )
            continue

        # ---------- Invalid class ----------
        try:
            class_id = int(parts[0])

            if class_id < 0 or class_id >= NUM_CLASSES:
                invalid_class.append(
                    (
                        label_path.name,
                        line_number,
                        class_id
                    )
                )

        except ValueError:
            invalid_class.append(
                (
                    label_path.name,
                    line_number,
                    parts[0]
                )
            )

# =====================================================
# REPORT
# =====================================================

print(f"Total images : {len(image_dict)}")
print(f"Total labels : {len(label_dict)}")
print()

# -----------------------------------------------------

print("=" * 60)
print("1. Missing Labels")
print("=" * 60)

if missing_labels:
    for item in missing_labels:
        print(item)
else:
    print("None")

print()

# -----------------------------------------------------

print("=" * 60)
print("2. Missing Images")
print("=" * 60)

if missing_images:
    for item in missing_images:
        print(item)
else:
    print("None")

print()

# -----------------------------------------------------

print("=" * 60)
print("3. Filename Mismatch")
print("=" * 60)

if filename_errors:
    for item in filename_errors:
        print(item)
else:
    print("None")

print()

# -----------------------------------------------------

print("=" * 60)
print("4. Empty Label Files")
print("=" * 60)

if empty_labels:
    for item in empty_labels:
        print(item)
else:
    print("None")

print()

# -----------------------------------------------------

print("=" * 60)
print("5. Invalid Label Format")
print("=" * 60)

if invalid_columns:

    for file_name, line_number, line in invalid_columns:
        print(
            f"{file_name} | line {line_number} -> {line}"
        )

else:
    print("None")

print()

# -----------------------------------------------------

print("=" * 60)
print("6. Invalid Class ID")
print("=" * 60)

if invalid_class:

    for file_name, line_number, class_id in invalid_class:
        print(
            f"{file_name} | line {line_number} -> class_id = {class_id}"
        )

else:
    print("None")

print()

# =====================================================
# SUMMARY
# =====================================================

print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Missing labels      : {len(missing_labels)}")
print(f"Missing images      : {len(missing_images)}")
print(f"Filename mismatch   : {len(filename_errors)}")
print(f"Empty labels        : {len(empty_labels)}")
print(f"Invalid columns     : {len(invalid_columns)}")
print(f"Invalid class IDs   : {len(invalid_class)}")

total_errors = (
    len(missing_labels)
    + len(missing_images)
    + len(empty_labels)
    + len(invalid_columns)
    + len(invalid_class)
)

print("-" * 60)

if total_errors == 0:
    print("Dataset check PASSED.")
else:
    print(f"Dataset check FAILED. Total errors: {total_errors}")
