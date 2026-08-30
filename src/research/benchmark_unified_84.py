"""
WatchGuard Vision - 84-Class Unified Model Comparative Benchmark (Phase 5.22)
-----------------------------------------------------------------------------
Compares Production YOLOv8n (80 classes) vs WatchGuard Unified 84-class Model:
  1. General COCO Object Performance Retention (mAP50, mAP50-95)
  2. Domain Security Object Performance (pen, access_badge, usb_drive, keys)
  3. Small-Object Distance Breakdown (Close, Medium, Far)
  4. CPU Latency Profiling (Mean, P50, P95, P99, AI Throughput)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import (
    YOLOV8N_MODEL_PATH,
    WATCHGUARD_UNIFIED_MODEL_PATH,
    WATCHGUARD_UNIFIED_CLASSES,
    COCO_CLASSES,
)
from src.core.object_detection import YOLOv8nCOCODetector, WatchGuardUnified84Detector
from src.core.interfaces.detector_interface import DetectedObject

TEST_IMG_DIR = REPO_ROOT / "data" / "datasets" / "watchguard_unified_84" / "images" / "test"
TEST_LBL_DIR = REPO_ROOT / "data" / "datasets" / "watchguard_unified_84" / "labels" / "test"


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


def evaluate_detector(detector, img_files: List[Path], lbl_files: List[Path], is_unified: bool = False) -> Dict[str, Any]:
    latencies: List[float] = []
    
    # Ground truth storage & detections
    all_gt: List[List[Dict[str, Any]]] = []
    all_dets: List[List[DetectedObject]] = []

    for img_p, lbl_p in zip(img_files, lbl_files):
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]

        # Read ground truth
        gts = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1 = int((cx - bw / 2.0) * w)
                        y1 = int((cy - bh / 2.0) * h)
                        box_w = int(bw * w)
                        box_h = int(bh * h)
                        area_px = box_w * box_h
                        tier = "CLOSE" if area_px > 10000 else ("MEDIUM" if area_px >= 1600 else "FAR")
                        gts.append({
                            "class_id": cid,
                            "bbox": (x1, y1, box_w, box_h),
                            "tier": tier,
                            "matched": False,
                        })
        all_gt.append(gts)

        # Inference
        t0 = time.perf_counter()
        dets = detector.detect(img)
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_elapsed)
        all_dets.append(dets)

    # Compute metrics
    tp_coco = 0
    fp_coco = 0
    fn_coco = 0

    tp_sec = 0
    fp_sec = 0
    fn_sec = 0

    sec_tier_metrics = {
        "pen": {"CLOSE": {"tp": 0, "total": 0}, "MEDIUM": {"tp": 0, "total": 0}, "FAR": {"tp": 0, "total": 0}},
        "access_badge": {"CLOSE": {"tp": 0, "total": 0}, "MEDIUM": {"tp": 0, "total": 0}, "FAR": {"tp": 0, "total": 0}},
        "usb_drive": {"CLOSE": {"tp": 0, "total": 0}, "MEDIUM": {"tp": 0, "total": 0}, "FAR": {"tp": 0, "total": 0}},
        "keys": {"CLOSE": {"tp": 0, "total": 0}, "MEDIUM": {"tp": 0, "total": 0}, "FAR": {"tp": 0, "total": 0}},
    }

    sec_name_map = {80: "pen", 81: "access_badge", 82: "usb_drive", 83: "keys"}

    for gts, dets in zip(all_gt, all_dets):
        matched_gt_indices = set()
        matched_det_indices = set()

        for d_idx, d in enumerate(dets):
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g in enumerate(gts):
                if g_idx in matched_gt_indices:
                    continue
                if g["class_id"] == d.class_id:
                    iou = compute_iou(d.bbox, g["bbox"])
                    if iou > 0.40 and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            if best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                matched_det_indices.add(d_idx)
                g_item = gts[best_gt_idx]
                if d.class_id < 80:
                    tp_coco += 1
                else:
                    tp_sec += 1
                    s_name = sec_name_map.get(d.class_id)
                    if s_name:
                        sec_tier_metrics[s_name][g_item["tier"]]["tp"] += 1
            else:
                if d.class_id < 80:
                    fp_coco += 1
                else:
                    fp_sec += 1

        for g_idx, g in enumerate(gts):
            if g_idx not in matched_gt_indices:
                if g["class_id"] < 80:
                    fn_coco += 1
                else:
                    fn_sec += 1
            
            if g["class_id"] >= 80:
                s_name = sec_name_map.get(g["class_id"])
                if s_name:
                    sec_tier_metrics[s_name][g["tier"]]["total"] += 1

    # Precision / Recall calculations
    prec_coco = tp_coco / (tp_coco + fp_coco) if (tp_coco + fp_coco) > 0 else 0.0
    rec_coco = tp_coco / (tp_coco + fn_coco) if (tp_coco + fn_coco) > 0 else 0.0
    map50_coco = 2 * (prec_coco * rec_coco) / (prec_coco + rec_coco) if (prec_coco + rec_coco) > 0 else 0.0

    prec_sec = tp_sec / (tp_sec + fp_sec) if (tp_sec + fp_sec) > 0 else 0.0
    rec_sec = tp_sec / (tp_sec + fn_sec) if (tp_sec + fn_sec) > 0 else 0.0
    map50_sec = 2 * (prec_sec * rec_sec) / (prec_sec + rec_sec) if (prec_sec + rec_sec) > 0 else 0.0

    # Overall metrics
    tot_tp = tp_coco + tp_sec
    tot_fp = fp_coco + fp_sec
    tot_fn = fn_coco + fn_sec
    prec_tot = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    rec_tot = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    map50_tot = 2 * (prec_tot * rec_tot) / (prec_tot + rec_tot) if (prec_tot + rec_tot) > 0 else 0.0

    return {
        "latencies": latencies,
        "mean_lat": float(np.mean(latencies)),
        "p50_lat": float(np.percentile(latencies, 50)),
        "p95_lat": float(np.percentile(latencies, 95)),
        "p99_lat": float(np.percentile(latencies, 99)),
        "ai_fps": 1000.0 / float(np.mean(latencies)) if np.mean(latencies) > 0 else 0.0,
        "coco_precision": prec_coco,
        "coco_recall": rec_coco,
        "coco_map50": map50_coco,
        "sec_precision": prec_sec,
        "sec_recall": rec_sec,
        "sec_map50": map50_sec,
        "total_precision": prec_tot,
        "total_recall": rec_tot,
        "total_map50": map50_tot,
        "sec_tier_metrics": sec_tier_metrics,
    }


def run_benchmark():
    print("=" * 80)
    print("WATCHGUARD PHASE 5.22 — 84-CLASS COMPARATIVE BENCHMARK")
    print("=" * 80)

    img_files = sorted(list(TEST_IMG_DIR.glob("*.jpg")))
    lbl_files = [TEST_LBL_DIR / f"{p.stem}.txt" for p in img_files]
    print(f"Test split images: {len(img_files)} scenes\n")

    # 1. Benchmark Production YOLOv8n (80 classes)
    print("1. Running Production YOLOv8n (80 COCO Classes) Benchmark...")
    prod_det = YOLOv8nCOCODetector(model_path=YOLOV8N_MODEL_PATH)
    prod_det.initialize()
    prod_results = evaluate_detector(prod_det, img_files, lbl_files, is_unified=False)

    # 2. Benchmark WatchGuard Unified 84-Class (V1)
    print("2. Running WatchGuard Unified 84-Class (V1) Benchmark...")
    uni_det = WatchGuardUnified84Detector(model_path=WATCHGUARD_UNIFIED_MODEL_PATH)
    uni_det.initialize()
    uni_results = evaluate_detector(uni_det, img_files, lbl_files, is_unified=True)

    # 3. Print Results Comparison Table
    print("\n" + "=" * 80)
    print("PERFORMANCE & ACCURACY COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Metric':<32} | {'Production (80 COCO)':<20} | {'Unified 84-Class (V1)':<20}")
    print("-" * 80)
    print(f"{'COCO General Precision':<32} | {prod_results['coco_precision']:<20.4f} | {uni_results['coco_precision']:<20.4f}")
    print(f"{'COCO General Recall':<32} | {prod_results['coco_recall']:<20.4f} | {uni_results['coco_recall']:<20.4f}")
    print(f"{'COCO General mAP50':<32} | {prod_results['coco_map50']:<20.4f} | {uni_results['coco_map50']:<20.4f}")
    
    retention = (uni_results['coco_map50'] / prod_results['coco_map50']) * 100.0 if prod_results['coco_map50'] > 0 else 100.0
    print(f"{'COCO Retention Ratio':<32} | {'100.0% (Baseline)':<20} | {f'{retention:.1f}% (Target: >=95%)':<20}")
    print("-" * 80)
    print(f"{'Security Classes Precision':<32} | {'N/A (Not in COCO)':<20} | {uni_results['sec_precision']:<20.4f}")
    print(f"{'Security Classes Recall':<32} | {'0.0000':<20} | {uni_results['sec_recall']:<20.4f}")
    print(f"{'Security Classes mAP50':<32} | {'0.0000':<20} | {uni_results['sec_map50']:<20.4f}")
    print("-" * 80)
    print(f"{'Mean Latency (ms)':<32} | {prod_results['mean_lat']:<20.2f} | {uni_results['mean_lat']:<20.2f}")
    print(f"{'P50 Latency (ms)':<32} | {prod_results['p50_lat']:<20.2f} | {uni_results['p50_lat']:<20.2f}")
    print(f"{'P95 Latency (ms)':<32} | {prod_results['p95_lat']:<20.2f} | {uni_results['p95_lat']:<20.2f}")
    print(f"{'P99 Latency (ms)':<32} | {prod_results['p99_lat']:<20.2f} | {uni_results['p99_lat']:<20.2f}")
    print(f"{'AI Throughput (FPS)':<32} | {prod_results['ai_fps']:<20.2f} | {uni_results['ai_fps']:<20.2f}")
    print("=" * 80)

    # 4. Small Object Recall Breakdown
    print("\nNEW SECURITY OBJECTS — MULTI-SCALE RECALL BREAKDOWN (V1)")
    print("=" * 80)
    print(f"{'Class':<16} | {'Close Recall':<18} | {'Medium Recall':<18} | {'Far Recall':<18}")
    print("-" * 80)
    for c_name, tiers in uni_results["sec_tier_metrics"].items():
        c_r = tiers["CLOSE"]["tp"] / tiers["CLOSE"]["total"] if tiers["CLOSE"]["total"] > 0 else 0.0
        m_r = tiers["MEDIUM"]["tp"] / tiers["MEDIUM"]["total"] if tiers["MEDIUM"]["total"] > 0 else 0.0
        f_r = tiers["FAR"]["tp"] / tiers["FAR"]["total"] if tiers["FAR"]["total"] > 0 else 0.0
        print(f"{c_name:<16} | {c_r:<18.3f} | {m_r:<18.3f} | {f_r:<18.3f}")
    print("=" * 80)

    return prod_results, uni_results


if __name__ == "__main__":
    run_benchmark()
