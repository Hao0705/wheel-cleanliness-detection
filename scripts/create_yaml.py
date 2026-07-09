import os
import yaml

# =====================================================
# Thông tin dataset
# =====================================================

dataset_info = {
    "train": "./train/images",
    "val": "./valid/images",
    "test": "./test/images",

    # Đường dẫn tuyệt đối tới thư mục dataset
    "path": os.path.abspath("./dataset"),

    # Số lượng class
    "nc": 2,

    # Tên các class theo đúng thứ tự class_id
    "names": [
        "clean_wheel",
        "dirty_wheel"
    ]
}

# =====================================================
# Tạo file data.yaml
# =====================================================

yaml_filepath = "../dataset/data.yaml"

with open(yaml_filepath, "w", encoding="utf-8") as f:
    yaml.dump(
        dataset_info,
        f,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True
    )

print("=" * 50)
print("Created successfully!")
print(f"File: {os.path.abspath(yaml_filepath)}")
print("=" * 50)

