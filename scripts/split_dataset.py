from pathlib import Path
import shutil
import random

# ==========================
# CONFIG
# ==========================

DATASET_DIR = Path("../dataset")

IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

# Các định dạng ảnh hỗ trợ
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

# ==========================
# Lấy danh sách ảnh
# ==========================

image_files = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
]

random.seed(RANDOM_SEED)
random.shuffle(image_files)

total = len(image_files)

train_num = int(total * TRAIN_RATIO)
valid_num = int(total * VALID_RATIO)

train_images = image_files[:train_num]
valid_images = image_files[train_num:train_num + valid_num]
test_images = image_files[train_num + valid_num:]

print(f"Total : {total}")
print(f"Train : {len(train_images)}")
print(f"Valid : {len(valid_images)}")
print(f"Test  : {len(test_images)}")


# ==========================
# Hàm copy
# ==========================

def copy_dataset(image_list, split_name):

    img_dst = DATASET_DIR / split_name / "images"
    lbl_dst = DATASET_DIR / split_name / "labels"

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for img_path in image_list:

        label_path = LABEL_DIR / (img_path.stem + ".txt")

        shutil.copy2(img_path, img_dst / img_path.name)

        if label_path.exists():
            shutil.copy2(label_path, lbl_dst / label_path.name)
        else:
            print(f"Warning: Missing label -> {label_path.name}")


copy_dataset(train_images, "train")
copy_dataset(valid_images, "valid")
copy_dataset(test_images, "test")

print("\nDataset split completed!")
