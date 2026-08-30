"""
WatchGuard Vision - 3-Way Comparative Small-Asset Benchmark Runner
------------------------------------------------------------------
Phase 5.19C: Evaluates and compares on the EXACT same held-out test set:
  1. Production YOLOv8n COCO (Baseline)
  2. Custom YOLOv8n-WatchGuard-V1 (Phase 5.19B Prototype)
  3. Custom YOLOv8n-WatchGuard-V2 (Phase 5.19C Expanded Dataset Model)

Stratified by Distance: CLOSE, MEDIUM, FAR.
Specifically reports Mouse-specific recall across all three distance tiers.
"""

import os
import sys
import time
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

sys.path.insert(0, r"d:\Projects\watchguard_vision")

from config import YOLOV8N_MODEL_PATH
from src.research.small_asset_benchmark import StandaloneSmallAssetBenchmarkHarness, categorize_distance_tier

MODEL_V1_PATH = "data/models/yolo_watchguard_assets_v1.onnx"
MODEL_V2_PATH = "data/models/yolo_watchguard_assets_v2.onnx"
TARGET_CLASSES = ["person", "laptop", "cell_phone", "mouse"]

# Load all three ONNX models
net_prod = cv2.dnn.readNetFromONNX(str(YOLOV8N_MODEL_PATH))
net_v1 = cv2.dnn.readNetFromONNX(MODEL_V1_PATH)
net_v2 = cv2.dnn.readNetFromONNX(MODEL_V2_PATH)

def preprocess_letterbox(img: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dx = (target_size - nw) // 2
    dy = (target_size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy

def run_prod_inference(frame: np.ndarray, conf_thresh: float = 0.35, nms_thresh: float = 0.45):
    h_orig, w_orig = frame.shape[:2]
    t0 = time.perf_counter()
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
    net_prod.setInput(blob)
    outputs = net_prod.forward()
    lat_ms = (time.perf_counter() - t0) * 1000.0

    predictions = np.transpose(outputs[0])
    scale_x = w_orig / 640.0
    scale_y = h_orig / 640.0

    scores_matrix = predictions[:, 4:]
    max_scores = np.max(scores_matrix, axis=1)
    valid_mask = max_scores >= conf_thresh

    if not np.any(valid_mask):
        return [], lat_ms

    filt_preds = predictions[valid_mask]
    filt_scores = scores_matrix[valid_mask]
    class_ids_arr = np.argmax(filt_scores, axis=1)
    confidences_arr = max_scores[valid_mask]

    cx, cy, w, h = filt_preds[:, 0], filt_preds[:, 1], filt_preds[:, 2], filt_preds[:, 3]
    x = np.clip((cx - 0.5 * w) * scale_x, 0, w_orig - 1).astype(int)
    y = np.clip((cy - 0.5 * h) * scale_y, 0, h_orig - 1).astype(int)
    bw = np.clip(w * scale_x, 1, w_orig - x).astype(int)
    bh = np.clip(h * scale_y, 1, h_orig - y).astype(int)

    boxes = np.stack([x, y, bw, bh], axis=1).tolist()
    confidences = confidences_arr.astype(float).tolist()
    class_ids = class_ids_arr.astype(int).tolist()

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    coco_to_target = {0: 0, 63: 1, 67: 2, 64: 3}
    
    results = []
    if len(indices) > 0:
        for idx in indices:
            i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
            c_id = class_ids[i]
            if c_id in coco_to_target:
                t_id = coco_to_target[c_id]
                results.append({
                    "class_id": t_id,
                    "label": TARGET_CLASSES[t_id],
                    "confidence": round(confidences[i], 3),
                    "bbox": tuple(boxes[i]),
                })
    return results, lat_ms

def run_custom_model_inference(net: cv2.dnn.Net, frame: np.ndarray, conf_thresh: float = 0.35, nms_thresh: float = 0.45):
    h_orig, w_orig = frame.shape[:2]
    t0 = time.perf_counter()
    canvas, scale, dx, dy = preprocess_letterbox(frame, target_size=640)
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    lat_ms = (time.perf_counter() - t0) * 1000.0

    predictions = np.transpose(outputs[0]) # (8400, 8)
    scores_matrix = predictions[:, 4:]     # 4 classes
    max_scores = np.max(scores_matrix, axis=1)
    valid_mask = max_scores >= conf_thresh

    if not np.any(valid_mask):
        return [], lat_ms

    filt_preds = predictions[valid_mask]
    filt_scores = scores_matrix[valid_mask]
    class_ids_arr = np.argmax(filt_scores, axis=1)
    confidences_arr = max_scores[valid_mask]

    cx, cy, w, h = filt_preds[:, 0], filt_preds[:, 1], filt_preds[:, 2], filt_preds[:, 3]
    x = np.clip((cx - 0.5 * w - dx) / scale, 0, w_orig - 1).astype(int)
    y = np.clip((cy - 0.5 * h - dy) / scale, 0, h_orig - 1).astype(int)
    bw = np.clip(w / scale, 1, w_orig - x).astype(int)
    bh = np.clip(h / scale, 1, h_orig - y).astype(int)

    boxes = np.stack([x, y, bw, bh], axis=1).tolist()
    confidences = confidences_arr.astype(float).tolist()
    class_ids = class_ids_arr.astype(int).tolist()

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    
    results = []
    if len(indices) > 0:
        for idx in indices:
            i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
            t_id = class_ids[i]
            if 0 <= t_id < len(TARGET_CLASSES):
                results.append({
                    "class_id": t_id,
                    "label": TARGET_CLASSES[t_id],
                    "confidence": round(confidences[i], 3),
                    "bbox": tuple(boxes[i]),
                })
    return results, lat_ms


# Exact Held-out Test Suite (Unchanged from Phase 5.19B)
def get_held_out_test_suite() -> List[Dict[str, Any]]:
    suite = []
    # 1. Close-up Person + Laptop + Mouse (Desk Office Scenario)
    f1 = np.full((480, 640, 3), (170, 165, 160), dtype=np.uint8)
    cv2.rectangle(f1, (80, 50), (320, 420), (40, 40, 40), -1)
    cv2.rectangle(f1, (360, 200), (520, 380), (30, 30, 30), -1)
    cv2.circle(f1, (440, 290), 50, (60, 60, 60), -1)
    suite.append({
        "image_id": "test_close_desk_01",
        "frame": f1,
        "ground_truths": [
            {"class_id": 1, "label": "laptop", "bbox": (80, 50, 240, 370), "distance_tier": "CLOSE"},
            {"class_id": 3, "label": "mouse", "bbox": (360, 200, 160, 180), "distance_tier": "CLOSE"},
        ],
    })

    # 2. Medium Distance Laptop + Cell Phone + Mouse
    f2 = np.full((480, 640, 3), (190, 185, 180), dtype=np.uint8)
    cv2.rectangle(f2, (150, 120), (300, 300), (45, 45, 45), -1)
    cv2.rectangle(f2, (340, 220), (410, 320), (20, 20, 20), -1)
    cv2.ellipse(f2, (480, 260), (35, 25), 0, 0, 360, (30, 30, 30), -1)
    suite.append({
        "image_id": "test_med_workstation_02",
        "frame": f2,
        "ground_truths": [
            {"class_id": 1, "label": "laptop", "bbox": (150, 120, 150, 180), "distance_tier": "MEDIUM"},
            {"class_id": 2, "label": "cell_phone", "bbox": (340, 220, 70, 100), "distance_tier": "MEDIUM"},
            {"class_id": 3, "label": "mouse", "bbox": (445, 235, 70, 50), "distance_tier": "MEDIUM"},
        ],
    })

    # 3. Far Distance Perimeter (Small Mouse + Small Phone on Conference Table)
    f3 = np.full((480, 640, 3), (150, 145, 140), dtype=np.uint8)
    cv2.rectangle(f3, (200, 180), (300, 260), (50, 50, 50), -1)
    cv2.rectangle(f3, (340, 220), (375, 270), (25, 25, 25), -1)
    cv2.ellipse(f3, (420, 240), (16, 11), 0, 0, 360, (20, 20, 20), -1)
    suite.append({
        "image_id": "test_far_conf_03",
        "frame": f3,
        "ground_truths": [
            {"class_id": 1, "label": "laptop", "bbox": (200, 180, 100, 80), "distance_tier": "MEDIUM"},
            {"class_id": 2, "label": "cell_phone", "bbox": (340, 220, 35, 50), "distance_tier": "FAR"},
            {"class_id": 3, "label": "mouse", "bbox": (404, 229, 32, 22), "distance_tier": "FAR"},
        ],
    })

    # 4. Live Webcam Capture Frame
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, live_f = cap.read()
        cap.release()
        if ret and live_f is not None:
            suite.append({
                "image_id": "test_live_webcam_04",
                "frame": live_f,
                "ground_truths": [
                    {"class_id": 0, "label": "person", "bbox": (100, 50, 350, 400), "distance_tier": "CLOSE"},
                ],
            })

    return suite


def run_comprehensive_3way_benchmark():
    print("="*75)
    print("RUNNING 3-WAY COMPARATIVE BENCHMARK (PROD vs V1 vs V2)")
    print("="*75)

    test_suite = get_held_out_test_suite()

    h_prod = StandaloneSmallAssetBenchmarkHarness("Production_YOLOv8n_COCO", target_classes=TARGET_CLASSES)
    h_v1 = StandaloneSmallAssetBenchmarkHarness("Custom_YOLOv8n_WatchGuard_V1", target_classes=TARGET_CLASSES)
    h_v2 = StandaloneSmallAssetBenchmarkHarness("Custom_YOLOv8n_WatchGuard_V2", target_classes=TARGET_CLASSES)

    for item in test_suite:
        img_id = item["image_id"]
        frame = item["frame"]
        gts = item["ground_truths"]
        shape = frame.shape[:2]

        # 1. Production Baseline
        preds_p, lat_p = run_prod_inference(frame, conf_thresh=0.35)
        h_prod.add_frame_evaluation(img_id, shape, gts, preds_p, lat_p)

        # 2. V1 Prototype
        preds_1, lat_1 = run_custom_model_inference(net_v1, frame, conf_thresh=0.35)
        h_v1.add_frame_evaluation(img_id, shape, gts, preds_1, lat_1)

        # 3. V2 Model
        preds_2, lat_2 = run_custom_model_inference(net_v2, frame, conf_thresh=0.35)
        h_v2.add_frame_evaluation(img_id, shape, gts, preds_2, lat_2)

    # Compute summaries
    s_prod = h_prod.compute_summary_metrics()
    s_v1 = h_v1.compute_summary_metrics()
    s_v2 = h_v2.compute_summary_metrics()

    # Export all reports
    h_prod.export_json_and_csv()
    h_v1.export_json_and_csv()
    h_v2.export_json_and_csv()

    print("\n" + "="*85)
    print(f"{'METRIC':32s} | {'PROD YOLOv8n':15s} | {'WATCHGUARD V1':15s} | {'WATCHGUARD V2':15s}")
    print("="*85)
    print(f"{'Overall Precision':32s} | {s_prod['aggregate_metrics']['overall_precision']:<15.3f} | {s_v1['aggregate_metrics']['overall_precision']:<15.3f} | {s_v2['aggregate_metrics']['overall_precision']:<15.3f}")
    print(f"{'Overall Recall':32s} | {s_prod['aggregate_metrics']['overall_recall']:<15.3f} | {s_v1['aggregate_metrics']['overall_recall']:<15.3f} | {s_v2['aggregate_metrics']['overall_recall']:<15.3f}")
    print(f"{'Overall F1 Score':32s} | {s_prod['aggregate_metrics']['overall_f1']:<15.3f} | {s_v1['aggregate_metrics']['overall_f1']:<15.3f} | {s_v2['aggregate_metrics']['overall_f1']:<15.3f}")
    print(f"{'Close-Tier Recall (>10,000 px^2)':32s} | {s_prod['aggregate_metrics']['close_tier_recall']:<15.3f} | {s_v1['aggregate_metrics']['close_tier_recall']:<15.3f} | {s_v2['aggregate_metrics']['close_tier_recall']:<15.3f}")
    print(f"{'Medium-Tier Recall (1,600-10,000)':32s} | {s_prod['aggregate_metrics']['medium_tier_recall']:<15.3f} | {s_v1['aggregate_metrics']['medium_tier_recall']:<15.3f} | {s_v2['aggregate_metrics']['medium_tier_recall']:<15.3f}")
    print(f"{'Far-Tier Small Recall (<1,600 px^2)':32s} | {s_prod['aggregate_metrics']['ar_small_far_recall']:<15.3f} | {s_v1['aggregate_metrics']['ar_small_far_recall']:<15.3f} | {s_v2['aggregate_metrics']['ar_small_far_recall']:<15.3f}")
    print("-" * 85)
    print(f"{'Mouse Recall (Class 3)':32s} | {s_prod['class_level_metrics']['mouse']['recall']:<15.3f} | {s_v1['class_level_metrics']['mouse']['recall']:<15.3f} | {s_v2['class_level_metrics']['mouse']['recall']:<15.3f}")
    print(f"{'Mouse Precision':32s} | {s_prod['class_level_metrics']['mouse']['precision']:<15.3f} | {s_v1['class_level_metrics']['mouse']['precision']:<15.3f} | {s_v2['class_level_metrics']['mouse']['precision']:<15.3f}")
    print("-" * 85)
    print(f"{'Mean Latency (ms)':32s} | {s_prod['latency_metrics']['mean_ms']:<15.2f} | {s_v1['latency_metrics']['mean_ms']:<15.2f} | {s_v2['latency_metrics']['mean_ms']:<15.2f}")
    print(f"{'P50 Latency (ms)':32s} | {s_prod['latency_metrics']['p50_ms']:<15.2f} | {s_v1['latency_metrics']['p50_ms']:<15.2f} | {s_v2['latency_metrics']['p50_ms']:<15.2f}")
    print(f"{'P95 Latency (ms)':32s} | {s_prod['latency_metrics']['p95_ms']:<15.2f} | {s_v1['latency_metrics']['p95_ms']:<15.2f} | {s_v2['latency_metrics']['p95_ms']:<15.2f}")
    print(f"{'AI Frame Rate (FPS)':32s} | {s_prod['latency_metrics']['theoretical_fps']:<15.1f} | {s_v1['latency_metrics']['theoretical_fps']:<15.1f} | {s_v2['latency_metrics']['theoretical_fps']:<15.1f}")
    print("="*85)


if __name__ == "__main__":
    run_comprehensive_3way_benchmark()
