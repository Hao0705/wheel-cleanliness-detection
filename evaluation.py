import os
from collections import defaultdict

GT_DIR = "dataset/test/labels"
PRED_DIR = "./runs/detect/predict_result/labels_pred/labels"

NUM_CLASSES = 2


def load_labels(folder):
    """
    return:
        {
            class_id:
            [
                (image_name, x, y, w, h),
                ...
            ]
        }
    """
    data = defaultdict(list)

    for file in os.listdir(folder):

        if not file.endswith(".txt"):
            continue

        image_name = os.path.splitext(file)[0]

        with open(os.path.join(folder, file), "r") as f:

            for line in f:

                arr = line.strip().split()

                if len(arr) < 5:
                    continue

                cls = int(arr[0])

                x = float(arr[1])
                y = float(arr[2])
                w = float(arr[3])
                h = float(arr[4])

                data[cls].append(
                    (
                        image_name,
                        x,
                        y,
                        w,
                        h
                    )
                )

    return data


def iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa1 = x1 - w1 / 2
    ya1 = y1 - h1 / 2
    xa2 = x1 + w1 / 2
    ya2 = y1 + h1 / 2

    xb1 = x2 - w2 / 2
    yb1 = y2 - h2 / 2
    xb2 = x2 + w2 / 2
    yb2 = y2 + h2 / 2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)

    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area1 = w1 * h1
    area2 = w2 * h2

    union = area1 + area2 - inter

    if union == 0:
        return 0

    return inter / union


gt = load_labels(GT_DIR)
pred = load_labels(PRED_DIR)

aps = []

for cls in range(NUM_CLASSES):

    gt_list = gt[cls]
    pred_list = pred[cls]

    matched = [False] * len(gt_list)

    TP = 0
    FP = 0

    for p in pred_list:

        ok = False

        for i, g in enumerate(gt_list):

            if matched[i]:
                continue

            if p[0] != g[0]:
                continue

            score = iou(
                p[1:],
                g[1:]
            )

            if score >= 0.5:

                TP += 1
                matched[i] = True
                ok = True
                break

        if not ok:
            FP += 1

    FN = len(gt_list) - TP

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0

    ap = precision * recall

    aps.append(ap)

    print()

    print(
        f"num_GT: {len(gt_list)}, num_Pred: {len(pred_list)}"
    )

    print(
        f"tp={TP}, fp={FP}, fn={FN}"
    )

    print(
        f"precision={precision:.4f}, recall={recall:.4f}"
    )

    print(
        f"ap={ap:.4f}"
    )

print()
print("=" * 50)
print(f"mAP50: {sum(aps)/len(aps):.6f}")