"""
WatchGuard Vision - Dataset V2 Pre-Training Audit Script
--------------------------------------------------------
Phase 5.19C: Validates dataset integrity before training V2 model.
Verifies:
  - Total image and label counts across splits
  - Class distribution (0: person, 1: laptop, 2: cell_phone, 3: mouse)
  - Distance tier distribution (Close, Medium, Far) for small assets
  - Empty / background image distribution
  - Detection of severe class imbalance or corrupted label files
"""

import os
import glob
from collections import defaultdict
from typing import Dict, Any, List
import json

DATASET_ROOT = "data/datasets/watchguard_assets_v1"
CLASS_NAMES = {0: "person", 1: "laptop", 2: "cell_phone", 3: "mouse"}

def audit_dataset() -> Dict[str, Any]:
    print("="*70)
    print("WATCHGUARD DATASET V2 PRE-TRAINING AUDIT")
    print("="*70)

    report = {}
    splits = ["train", "val"]
    
    total_images_all = 0
    total_labels_all = 0
    class_counts_all = defaultdict(int)
    tier_counts_mouse = defaultdict(int)
    empty_images_count = 0

    for split in splits:
        img_dir = os.path.join(DATASET_ROOT, "images", split)
        lbl_dir = os.path.join(DATASET_ROOT, "labels", split)

        images = glob.glob(os.path.join(img_dir, "*.*"))
        labels = glob.glob(os.path.join(lbl_dir, "*.txt"))

        img_basenames = {os.path.splitext(os.path.basename(f))[0] for f in images}
        lbl_basenames = {os.path.splitext(os.path.basename(f))[0] for f in labels}

        missing_labels = img_basenames - lbl_basenames
        missing_images = lbl_basenames - img_basenames

        split_class_counts = defaultdict(int)
        split_tier_counts = {"CLOSE": 0, "MEDIUM": 0, "FAR": 0}
        split_empty = 0

        for lbl_file in labels:
            with open(lbl_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if not lines:
                split_empty += 1
                empty_images_count += 1
                continue

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                area_norm = w * h # in 640x480 space: (w*640)*(h*480)
                area_px = (w * 640) * (h * 480)

                split_class_counts[cls_id] += 1
                class_counts_all[cls_id] += 1

                if cls_id == 3: # mouse
                    if area_px > 10000:
                        tier = "CLOSE"
                    elif area_px >= 1600:
                        tier = "MEDIUM"
                    else:
                        tier = "FAR"
                    split_tier_counts[tier] += 1
                    tier_counts_mouse[tier] += 1

        total_images_all += len(images)
        total_labels_all += len(labels)

        report[split] = {
            "image_count": len(images),
            "label_file_count": len(labels),
            "missing_label_files": len(missing_labels),
            "missing_image_files": len(missing_images),
            "empty_background_images": split_empty,
            "class_distribution": {CLASS_NAMES[k]: v for k, v in split_class_counts.items()},
            "mouse_distance_distribution": split_tier_counts,
        }

        print(f"\n--- Split: {split.upper()} ---")
        print(f"  Images: {len(images)}, Labels: {len(labels)}, Empty background images: {split_empty}")
        print(f"  Class breakdown:")
        for cid, name in CLASS_NAMES.items():
            print(f"    {cid} ({name:12s}): {split_class_counts[cid]:4d} instances")
        print(f"  Mouse Distance Stratification:")
        print(f"    CLOSE  (>10,000 px^2):     {split_tier_counts['CLOSE']:4d}")
        print(f"    MEDIUM (1,600-10,000 px^2):{split_tier_counts['MEDIUM']:4d}")
        print(f"    FAR    (<1,600 px^2):      {split_tier_counts['FAR']:4d}")

    # Check for imbalance or failure conditions
    audit_passed = True
    warnings = []

    if report["train"]["missing_label_files"] > 0 or report["train"]["missing_image_files"] > 0:
        audit_passed = False
        warnings.append("Mismatched image/label file pairs found.")

    if class_counts_all[3] < 300:
        audit_passed = False
        warnings.append(f"Mouse training instances ({class_counts_all[3]}) below minimum target of 300.")

    # Check distance balance
    m_close = tier_counts_mouse["CLOSE"]
    m_med = tier_counts_mouse["MEDIUM"]
    m_far = tier_counts_mouse["FAR"]
    if min(m_close, m_med, m_far) < 50:
        audit_passed = False
        warnings.append(f"Severe distance tier imbalance in mouse samples: Close={m_close}, Med={m_med}, Far={m_far}")

    print("\n" + "="*70)
    print(f"AUDIT STATUS: {'PASSED [READY FOR TRAINING]' if audit_passed else 'FAILED'}")
    if warnings:
        print("Warnings/Errors:")
        for w in warnings:
            print("  -", w)
    print("="*70)

    report["overall"] = {
        "total_images": total_images_all,
        "total_annotations": sum(class_counts_all.values()),
        "class_totals": {CLASS_NAMES[k]: v for k, v in class_counts_all.items()},
        "mouse_tier_totals": dict(tier_counts_mouse),
        "audit_passed": audit_passed,
    }

    with open("data/research/dataset_v2_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    audit_dataset()
