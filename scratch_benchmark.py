import time
import cv2
import numpy as np
import os
from src.core.object_detection import ObjectDetectionEngine
from src.core.person_recognition import PersonRecognitionEngine
from src.core.context_engine import ContextEngine

print('=== PHASE 5.8 YOLOv8 & PIPELINE OPTIMIZATION BENCHMARK ===')

# Generate test synthetic frame (640x480)
frame = np.full((480, 640, 3), 128, dtype=np.uint8)
cv2.rectangle(frame, (100, 100), (300, 400), (200, 180, 160), -1) # Person body
cv2.circle(frame, (200, 150), 40, (220, 200, 180), -1) # Face
cv2.rectangle(frame, (220, 250), (260, 320), (30, 30, 30), -1) # Cell phone

engine = ObjectDetectionEngine()
engine.initialize()

person_engine = PersonRecognitionEngine()
person_engine.initialize()

context_engine = ContextEngine()

print(f'YOLOv8 Engine Ready: {engine.is_ready}')

# Warmup
for _ in range(5):
    _ = engine.process_frame(frame)

# 1. Benchmark 640x640 ONNX Inference (Default)
latencies_640 = []
t_start = time.perf_counter()
for _ in range(60):
    t0 = time.perf_counter()
    res = engine.process_frame(frame)
    latencies_640.append((time.perf_counter() - t0) * 1000.0)
t_end = time.perf_counter()

mean_640 = np.mean(latencies_640)
p95_640 = np.percentile(latencies_640, 95)
fps_640 = len(latencies_640) / (t_end - t_start)
print(f'Option 1 (YOLOv8 640x640 Input): Mean={mean_640:.2f}ms, p95={p95_640:.2f}ms, AI FPS={fps_640:.2f}')

# Full AI Pipeline Latency Benchmark (YOLO + Face + PAD + Context)
full_ai_latencies = []
for _ in range(60):
    t0 = time.perf_counter()
    obj_res = engine.process_frame(frame)
    p_res = person_engine.process_frame(frame, detected_objects=obj_res.get('objects', []))
    dec = context_engine.evaluate_context(p_res, obj_res, camera_id=0, frame_shape=frame.shape)
    full_ai_latencies.append((time.perf_counter() - t0) * 1000.0)

mean_full = np.mean(full_ai_latencies)
p95_full = np.percentile(full_ai_latencies, 95)
ai_fps_full = 1000.0 / mean_full
print(f'Full AI Pipeline (YOLO + Face + PAD + Context): Mean={mean_full:.2f}ms, p95={p95_full:.2f}ms, AI FPS={ai_fps_full:.2f}')
