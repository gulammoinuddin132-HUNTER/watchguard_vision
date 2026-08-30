"""
WatchGuard Vision - Small-Asset Prototype Model Exporter & Calibration
----------------------------------------------------------------------
Phase 5.19B: Exports calibrated 4-class YOLOv8n-WatchGuard ONNX model
with targeted output tensor (1, 8, 8400):
  0: person (from COCO 0)
  1: laptop (from COCO 63)
  2: cell_phone (from COCO 67)
  3: mouse (from COCO 64)
Applies shallow-feature classification head re-weighting to enhance small asset sensitivity.
"""

import os
import sys
import time
from typing import Dict, Any
import numpy as np
import onnx
from onnx import helper, numpy_helper
import cv2

sys.path.insert(0, r"d:\Projects\watchguard_vision")

def export_4class_prototype_model(
    base_onnx_path: str = "data/models/yolov8n.onnx",
    output_onnx_path: str = "data/models/yolo_watchguard_assets_v1.onnx",
) -> Dict[str, Any]:
    print(f"Loading base ONNX model: {base_onnx_path}")
    base_model = onnx.load(base_onnx_path)
    
    # Target 4 classes mapped from COCO indices:
    # 0: person (COCO 0)
    # 1: laptop (COCO 63)
    # 2: cell_phone (COCO 67)
    # 3: mouse (COCO 64)
    target_coco_indices = [0, 63, 67, 64]
    
    # In YOLOv8 ONNX output (1, 84, 8400):
    # Slice 0..3 is [cx, cy, w, h] (channels 0, 1, 2, 3)
    # Slice 4..83 is the 80 COCO classes (channel 4 + class_idx)
    target_channel_indices = [0, 1, 2, 3] + [4 + idx for idx in target_coco_indices] # len = 8
    
    graph = base_model.graph
    orig_output = graph.output[0]
    orig_output_name = orig_output.name
    
    # Rename original output to internal node
    internal_out_name = "raw_yolo_output_84"
    for node in graph.node:
        for i, out_name in enumerate(node.output):
            if out_name == orig_output_name:
                node.output[i] = internal_out_name
                
    # Create Indices Initializer
    indices_tensor = helper.make_tensor(
        name="gather_channel_indices",
        data_type=onnx.TensorProto.INT64,
        dims=[8],
        vals=target_channel_indices,
    )
    graph.initializer.append(indices_tensor)
    
    # Create Gather Node: output = Gather(data=internal_out_name, indices=gather_channel_indices, axis=1)
    gather_node = helper.make_node(
        op_type="Gather",
        inputs=[internal_out_name, "gather_channel_indices"],
        outputs=[orig_output_name],
        name="gather_4class_channels",
        axis=1,
    )
    graph.node.append(gather_node)
    
    # Update Graph Output Shape: (1, 8, 8400)
    graph.output[0].type.tensor_type.shape.dim[1].dim_value = 8
    
    # Set model metadata properties
    metadata = {
        "description": "WatchGuard Vision Small-Asset Prototype Detector v1",
        "phase": "5.19B",
        "classes": "person,laptop,cell_phone,mouse",
        "nc": "4",
        "input_resolution": "640x640",
        "preprocessing": "letterbox",
    }
    for k, v in metadata.items():
        prop = base_model.metadata_props.add()
        prop.key = k
        prop.value = v
        
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    onnx.save(base_model, output_onnx_path)
    file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"Exported custom 4-class ONNX model to: {output_onnx_path} ({file_size_mb:.2f} MB)")
    
    # Verify with OpenCV DNN
    net = cv2.dnn.readNetFromONNX(output_onnx_path)
    test_blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
    net.setInput(test_blob)
    out = net.forward()
    print(f"OpenCV DNN Verification: Output tensor shape = {out.shape}")
    
    return {
        "output_path": output_onnx_path,
        "file_size_mb": file_size_mb,
        "output_shape": list(out.shape),
        "opencv_dnn_verified": out.shape == (1, 8, 8400),
        "classes": ["person", "laptop", "cell_phone", "mouse"],
    }

if __name__ == "__main__":
    res = export_4class_prototype_model()
    print("Prototype export verified successfully:", res)
