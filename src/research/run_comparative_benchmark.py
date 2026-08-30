"""
WatchGuard Vision - Comparative Small-Asset Benchmark Runner
------------------------------------------------------------
Phase 5.19B: Evaluates and compares:
  1. Baseline YOLOv8n COCO (Production Model with Direct Stretch)
  2. Custom WatchGuard 4-Class Prototype (with Aspect-Preserving Letterboxing)
Stratified across Close, Medium, and Far tiers for the 4 initial prototype classes.
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, r"d:\Projects\watchguard_vision")

from config import YOLOV8N_MODEL_PATH, COCO_CLASSES
from src.research.small_asset_benchmark import StandaloneSmallAssetBenchmarkHarness

PROTOTYPE_MODEL_PATH = "data/models/yolo_watchguard_assets_v1.onnx"
TARGET_CLASSES = ["person", "laptop", "cell_phone", "mouse"]

# Load ONNX nets
net_baseline = cv2.dnn.readNetFromONNX(str(YOLOV8N_MODEL_PATH))
net_prototype = cv2.dnn.readNetFromONNX(PROTOTYPE_MODEL_PATH)

def preprocess_letterbox(img: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, int, int]:
    """Aspect-preserving letterbox with 114 gray padding."""
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dx = (target_size - nw) // 2
    dy = (target_size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def run_baseline_inference(frame: np.ndarray, conf_thresh: float = 0.40, nms_thresh: float = 0.45):
    """Baseline YOLOv8n inference with direct stretching."""
    h_orig, w_orig = frame.shape[:2]
    t0 = time.perf_counter()
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
    net_baseline.setInput(blob)
    outputs = net_baseline.forward()
    lat_ms = (time.perf_counter() - t0) * 1000.0

    predictions = np.transpose(outputs[0])  # (8400, 84)
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
    
    # Map COCO classes back to our 4 target classes:
    # COCO 0 -> 0 (person), COCO 63 -> 1 (laptop), COCO 67 -> 2 (cell_phone), COCO 64 -> 3 (mouse)
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


def run_prototype_inference(frame: np.ndarray, conf_thresh: float = 0.40, nms_thresh: float = 0.45):
    """Custom WatchGuard prototype inference with aspect-preserving letterbox."""
    h_orig, w_orig = frame.shape[:2]
    t0 = time.perf_counter()
    canvas, scale, dx, dy = preprocess_letterbox(frame, target_size=640)
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
    net_prototype.setInput(blob)
    outputs = net_prototype.forward()
    lat_ms = (time.perf_counter() - t0) * 1000.0

    predictions = np.transpose(outputs[0])  # (8400, 8)
    scores_matrix = predictions[:, 4:]      # 4 target classes
    max_scores = np.max(scores_matrix, axis=1)
    valid_mask = max_scores >= conf_thresh

    if not np.any(valid_mask):
        return [], lat_ms

    filt_preds = predictions[valid_mask]
    filt_scores = scores_matrix[valid_mask]
    class_ids_arr = np.argmax(filt_scores, axis=1)
    confidences_arr = max_scores[valid_mask]

    cx, cy, w, h = filt_preds[:, 0], filt_preds[:, 1], filt_preds[:, 2], filt_preds[:, 3]
    # Un-pad and scale back
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


# Generate test evaluation suite
def create_test_evaluation_suite() -> List[Dict[str, Any]]:
    """Generates held-out multi-tier test scenarios covering Close, Medium, Far tiers."""
    suite = []
    
    # 1. Close-up Person + Laptop + Mouse (Desk Office Scenario)
    f1 = np.full((480, 640, 3), (170, 165, 160), dtype=np.uint8)
    cv2.rectangle(f1, (80, 50), (320, 420), (40, 40, 40), -1) # Laptop (Close)
    cv2.rectangle(f1, (360, 200), (520, 380), (30, 30, 30), -1) # Mouse (Close: 160x180)
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
    cv2.rectangle(f2, (150, 120), (300, 300), (45, 45, 45), -1) # Laptop (Medium: 150x180)
    cv2.rectangle(f2, (340, 220), (410, 320), (20, 20, 20), -1) # Phone (Medium: 70x100)
    cv2.ellipse(f2, (480, 260), (35, 25), 0, 0, 360, (30, 30, 30), -1) # Mouse (Medium: 70x50 = 3500px^2)
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
    cv2.rectangle(f3, (200, 180), (300, 260), (50, 50, 50), -1) # Laptop (Medium: 100x80)
    cv2.rectangle(f3, (340, 220), (375, 270), (25, 25, 25), -1) # Phone (Far: 35x50 = 1750px^2)
    cv2.ellipse(f3, (420, 240), (16, 11), 0, 0, 360, (20, 20, 20), -1) # Mouse (Far: 32x22 = 704px^2)
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


def execute_benchmark_comparison():
    print("="*70)
    print("RUNNING COMPARATIVE BENCHMARK (BASELINE vs CUSTOM PROTOTYPE)")
    print("="*70)

    test_suite = create_test_evaluation_suite()
    print(f"Loaded {len(test_suite)} test scenarios.")

    harness_base = StandaloneSmallAssetBenchmarkHarness(
        model_name="Baseline_YOLOv8n_COCO",
        target_classes=TARGET_CLASSES,
        output_dir="data/research",
    )
    
    harness_proto = StandaloneSmallAssetBenchmarkHarness(
        model_name="Custom_YOLOv8n_WatchGuard_v1",
        target_classes=TARGET_CLASSES,
        output_dir="data/research",
    )

    for item in test_suite:
        img_id = item["image_id"]
        frame = item["frame"]
        gts = item["ground_truths"]
        shape = frame.shape[:2]

        # 1. Baseline Run
        preds_base, lat_base = run_baseline_inference(frame, conf_thresh=0.35)
        harness_base.add_frame_evaluation(
            image_id=img_id,
            frame_shape=shape,
            ground_truths=gts,
            predictions=preds_base,
            latency_ms=lat_base,
        )

        # 2. Prototype Run
        preds_proto, lat_proto = run_prototype_inference(frame, conf_thresh=0.35)
        harness_proto.add_frame_evaluation(
            image_id=img_id,
            frame_shape=shape,
            ground_truths=gts,
            predictions=preds_proto,
            latency_ms=lat_proto,
        )

    # Export both
    j_base, c_base = harness_base.export_json_and_csv()
    j_proto, c_proto = harness_proto.export_json_and_csv()

    summary_base = harness_base.compute_summary_metrics()
    summary_proto = harness_proto.compute_summary_metrics()

    print("\n" + "="*70)
    print("BENCHMARK COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':30s} | {'Baseline YOLOv8n COCO':22s} | {'Custom WatchGuard v1':22s}")
    print("-" * 80)
    print(f"{'Total Ground Truths':30s} | {summary_base['aggregate_metrics']['total_ground_truths']:<22d} | {summary_proto['aggregate_metrics']['total_ground_truths']:<22d}")
    print(f"{'True Positives':30s} | {summary_base['aggregate_metrics']['true_positives']:<22d} | {summary_proto['aggregate_metrics']['true_positives']:<22d}")
    print(f"{'Overall Precision':30s} | {summary_base['aggregate_metrics']['overall_precision']:<22.3f} | {summary_proto['aggregate_metrics']['overall_precision']:<22.3f}")
    print(f"{'Overall Recall':30s} | {summary_base['aggregate_metrics']['overall_recall']:<22.3f} | {summary_proto['aggregate_metrics']['overall_recall']:<22.3f}")
    print(f"{'Overall F1 Score':30s} | {summary_base['aggregate_metrics']['overall_f1']:<22.3f} | {summary_proto['aggregate_metrics']['overall_f1']:<22.3f}")
    print(f"{'Close-Tier Recall':30s} | {summary_base['aggregate_metrics']['close_tier_recall']:<22.3f} | {summary_proto['aggregate_metrics']['close_tier_recall']:<22.3f}")
    print(f"{'Medium-Tier Recall':30s} | {summary_base['aggregate_metrics']['medium_tier_recall']:<22.3f} | {summary_proto['aggregate_metrics']['medium_tier_recall']:<22.3f}")
    print(f"{'Far-Tier Small Recall (AR_small)':30s} | {summary_base['aggregate_metrics']['ar_small_far_recall']:<22.3f} | {summary_proto['aggregate_metrics']['ar_small_far_recall']:<22.3f}")
    print(f"{'Mean Latency (ms)':30s} | {summary_base['latency_metrics']['mean_ms']:<22.2f} | {summary_proto['latency_metrics']['mean_ms']:<22.2f}")
    print(f"{'P50 Latency (ms)':30s} | {summary_base['latency_metrics']['p50_ms']:<22.2f} | {summary_proto['latency_metrics']['p50_ms']:<22.2f}")
    print(f"{'P95 Latency (ms)':30s} | {summary_base['latency_metrics']['p95_ms']:<22.2f} | {summary_proto['latency_metrics']['p95_ms']:<22.2f}")
    print(f"{'Theoretical AI FPS':30s} | {summary_base['latency_metrics']['theoretical_fps']:<22.1f} | {summary_proto['latency_metrics']['theoretical_fps']:<22.1f}")
    print("-" * 80)
    print(f"Results exported to:\n  {j_base}\n  {j_proto}")


if __name__ == "__main__":
    execute_benchmark_comparison()
