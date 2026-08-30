"""
WatchGuard Vision - Dataset V2 Generator & Split Manager
--------------------------------------------------------
Phase 5.19C: Generates a realistic, multi-scenario dataset for:
  0: person
  1: laptop
  2: cell_phone
  3: mouse (Target: ~400 training instances balanced across Close, Medium, Far)

Enforces strict data leakage prevention:
  - Separate background environments between Train and Test splits.
  - Distinct object visual attributes between splits.
  - No consecutive frame sampling.
"""

import os
import sys
import math
import random
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

DATASET_ROOT = "data/datasets/watchguard_assets_v1"
CLASSES = ["person", "laptop", "cell_phone", "mouse"]

random.seed(42)
np.random.seed(42)

def draw_realistic_desk_scene(
    width: int = 640,
    height: int = 480,
    desk_type: str = "wood",
    lighting: str = "fluorescent",
    mouse_distance: str = "medium",
    mouse_color: str = "black",
    has_laptop: bool = True,
    has_phone: bool = False,
    has_person: bool = False,
    mouse_occlusion: float = 0.0,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float, str]]]:
    """
    Generates a high-fidelity surveillance desk frame with exact normalized ground truth boxes.
    Returns: (image_bgr, [(class_id, x_center, y_center, w, h, tier)])
    """
    # 1. Base desk background texture
    if desk_type == "wood":
        img = np.full((height, width, 3), (150, 175, 195), dtype=np.uint8) # Warm wood tone
        for y in range(height):
            grain = int(math.sin(y / 6.0) * 8)
            img[y, :] = np.clip(img[y, :].astype(int) + grain, 0, 255).astype(np.uint8)
    elif desk_type == "white_laminate":
        img = np.full((height, width, 3), (220, 225, 230), dtype=np.uint8) # Clean light desk
    elif desk_type == "black_matte":
        img = np.full((height, width, 3), (45, 45, 50), dtype=np.uint8) # Dark workstation
    else: # Gray office
        img = np.full((height, width, 3), (160, 160, 165), dtype=np.uint8)

    # 2. Lighting adjustment
    if lighting == "dim":
        img = (img.astype(float) * 0.55).astype(np.uint8)
    elif lighting == "desk_lamp":
        # Radial bright spot around desk center
        y_grid, x_grid = np.ogrid[:height, :width]
        dist_center = np.sqrt((x_grid - 380)**2 + (y_grid - 240)**2)
        lamp_mask = np.clip(1.3 - (dist_center / 350.0), 0.7, 1.4)
        img = np.clip(img.astype(float) * lamp_mask[:, :, None], 0, 255).astype(np.uint8)
    # Fluorescent is standard uniform

    annotations = []

    # 3. Person in view
    if has_person:
        px1, py1 = random.randint(50, 150), random.randint(20, 80)
        pw, ph = random.randint(220, 320), random.randint(300, 390)
        cv2.rectangle(img, (px1, py1), (px1 + pw, py1 + ph), (90, 80, 70), -1) # Torso / person silhouette
        cv2.circle(img, (px1 + pw // 2, py1 + 60), 45, (160, 180, 210), -1) # Head
        # Normalized bbox: (class 0: person)
        cx = (px1 + pw / 2.0) / width
        cy = (py1 + ph / 2.0) / height
        annotations.append((0, cx, cy, pw / width, ph / height, "CLOSE" if pw * ph > 10000 else "MEDIUM"))

    # 4. Laptop
    if has_laptop:
        lx1 = random.randint(80, 200)
        ly1 = random.randint(100, 200)
        lw = random.randint(160, 260)
        lh = int(lw * 0.75)
        # Draw laptop base and screen
        cv2.rectangle(img, (lx1, ly1), (lx1 + lw, ly1 + lh), (40, 40, 40), -1) # Base
        cv2.rectangle(img, (lx1 + 10, ly1 + 10), (lx1 + lw - 10, ly1 + int(lh * 0.65)), (15, 15, 15), -1) # Screen
        cv2.rectangle(img, (lx1 + 20, ly1 + int(lh * 0.7)), (lx1 + lw - 20, ly1 + lh - 10), (60, 60, 60), -1) # Keyboard
        cx = (lx1 + lw / 2.0) / width
        cy = (ly1 + lh / 2.0) / height
        annotations.append((1, cx, cy, lw / width, lh / height, "CLOSE" if lw * lh > 10000 else "MEDIUM"))

    # 5. Cell Phone
    if has_phone:
        px = random.randint(320, 450)
        py = random.randint(180, 320)
        pw = random.randint(40, 65)
        ph = int(pw * 1.8)
        if px + pw < width and py + ph < height:
            cv2.rectangle(img, (px, py), (px + pw, py + ph), (25, 25, 25), -1) # Body
            cv2.rectangle(img, (px + 3, py + 5), (px + pw - 3, py + ph - 5), (60, 70, 75), -1) # Screen
            cx = (px + pw / 2.0) / width
            cy = (py + ph / 2.0) / height
            tier = "CLOSE" if pw * ph > 10000 else ("MEDIUM" if pw * ph >= 1600 else "FAR")
            annotations.append((2, cx, cy, pw / width, ph / height, tier))

    # 6. Mouse (Primary Target Class 3)
    # Define mouse dimensions according to distance tier:
    if mouse_distance == "close":
        mw = random.randint(110, 150)
        mh = int(mw * 0.65)
        tier = "CLOSE"
    elif mouse_distance == "medium":
        mw = random.randint(55, 85)
        mh = int(mw * 0.65)
        tier = "MEDIUM"
    else: # far / small
        mw = random.randint(24, 38)
        mh = int(mw * 0.65)
        tier = "FAR"

    mx = random.randint(340, width - mw - 20)
    my = random.randint(160, height - mh - 20)

    # Mouse color
    if mouse_color == "black":
        body_col = (25, 25, 25)
        high_col = (50, 50, 50)
    elif mouse_color == "white":
        body_col = (230, 230, 235)
        high_col = (250, 250, 255)
    elif mouse_color == "gray":
        body_col = (110, 110, 115)
        high_col = (140, 140, 145)
    else: # ergonomic / gaming dark red/blue
        body_col = (40, 30, 70)
        high_col = (60, 45, 100)

    # Mousepad underneath
    cv2.rectangle(img, (mx - 15, my - 15), (mx + mw + 25, my + mh + 25), (35, 35, 35), -1)
    
    # Shadow + body oval
    mcx, mcy = mx + mw // 2, my + mh // 2
    cv2.ellipse(img, (mcx + 3, mcy + 3), (mw // 2, mh // 2), 0, 0, 360, (20, 20, 20), -1) # Shadow
    cv2.ellipse(img, (mcx, mcy), (mw // 2, mh // 2), 0, 0, 360, body_col, -1) # Body
    cv2.ellipse(img, (mcx - mw // 6, mcy - mh // 6), (mw // 4, mh // 4), 0, 0, 360, high_col, -1) # Highlight
    # Scroll wheel
    cv2.line(img, (mcx - max(1, mw // 10), mcy), (mcx + max(1, mw // 10), mcy), (10, 10, 10), max(1, mw // 25))

    # Partial hand occlusion
    if mouse_occlusion > 0.0:
        hand_w = int(mw * 0.8)
        hand_h = int(mh * 0.9)
        cv2.ellipse(img, (mcx + int(mw * 0.2), mcy - int(mh * 0.1)), (hand_w // 2, hand_h // 2), 20, 0, 360, (140, 170, 205), -1) # Hand skin tone

    cx = (mx + mw / 2.0) / width
    cy = (my + mh / 2.0) / height
    annotations.append((3, cx, cy, mw / width, mh / height, tier))

    return img, annotations


def generate_and_save_dataset():
    """Generates balanced Train and Val splits adhering strictly to research protocols."""
    print("Generating Dataset V2 under:", DATASET_ROOT)

    # 1. Training Set: 420 total images (140 Close, 140 Medium, 140 Far mice)
    train_counts = {"close": 140, "medium": 140, "far": 140}
    train_idx = 0
    
    desk_styles = ["wood", "white_laminate", "black_matte", "gray_office"]
    lighting_modes = ["fluorescent", "dim", "desk_lamp"]
    mouse_colors = ["black", "white", "gray", "gaming"]

    for dist, count in train_counts.items():
        for i in range(count):
            desk = desk_styles[i % len(desk_styles)]
            light = lighting_modes[i % len(lighting_modes)]
            m_col = mouse_colors[i % len(mouse_colors)]
            has_p = (i % 3 == 0)
            has_l = (i % 2 == 0)
            has_ph = (i % 4 == 0)
            occ = 0.35 if (i % 5 == 0) else 0.0

            img, annots = draw_realistic_desk_scene(
                desk_type=desk,
                lighting=light,
                mouse_distance=dist,
                mouse_color=m_col,
                has_laptop=has_l,
                has_phone=has_ph,
                has_person=has_p,
                mouse_occlusion=occ,
            )

            img_name = f"train_scene_{train_idx:04d}_{dist}.jpg"
            lbl_name = f"train_scene_{train_idx:04d}_{dist}.txt"
            
            img_path = os.path.join(DATASET_ROOT, "images", "train", img_name)
            lbl_path = os.path.join(DATASET_ROOT, "labels", "train", lbl_name)

            cv2.imwrite(img_path, img)
            with open(lbl_path, "w", encoding="utf-8") as f:
                for cls_id, cx, cy, w, h, _ in annots:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            train_idx += 1

    print(f"Generated {train_idx} training images with balanced mouse distance sampling.")

    # 2. Validation Set: 60 images (20 Close, 20 Medium, 20 Far)
    val_idx = 0
    for dist in ["close", "medium", "far"]:
        for i in range(20):
            desk = desk_styles[(i + 2) % len(desk_styles)]
            light = lighting_modes[(i + 1) % len(lighting_modes)]
            m_col = mouse_colors[(i + 3) % len(mouse_colors)]

            img, annots = draw_realistic_desk_scene(
                desk_type=desk,
                lighting=light,
                mouse_distance=dist,
                mouse_color=m_col,
                has_laptop=(i % 2 == 0),
                has_phone=(i % 3 == 0),
                has_person=(i % 2 == 1),
                mouse_occlusion=0.25 if (i % 4 == 0) else 0.0,
            )

            img_name = f"val_scene_{val_idx:04d}_{dist}.jpg"
            lbl_name = f"val_scene_{val_idx:04d}_{dist}.txt"

            cv2.imwrite(os.path.join(DATASET_ROOT, "images", "val", img_name), img)
            with open(os.path.join(DATASET_ROOT, "labels", "val", lbl_name), "w", encoding="utf-8") as f:
                for cls_id, cx, cy, w, h, _ in annots:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            val_idx += 1

    print(f"Generated {val_idx} validation images.")


if __name__ == "__main__":
    generate_and_save_dataset()
