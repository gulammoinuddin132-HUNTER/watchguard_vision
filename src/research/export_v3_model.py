"""
WatchGuard Vision - Export YOLOv8n/11n WatchGuard Assets V3 Model
-----------------------------------------------------------------
Phase 5.19D: Exports data/models/yolo_watchguard_assets_v3.onnx for production/custom switching.
Target classes:
  0: person
  1: laptop
  2: cell_phone
  3: mouse
"""

import os
import sys
import numpy as np
import onnx
from onnx import helper
import cv2

def export_v3_model():
    base_onnx = "data/models/yolov8n.onnx"
    output_onnx = "data/models/yolo_watchguard_assets_v3.onnx"
    
    print(f"Loading base model: {base_onnx}")
    model = onnx.load(base_onnx)
    graph = model.graph
    
    # Map COCO 0 (person), 63 (laptop), 67 (cell phone), 64 (mouse)
    target_coco_indices = [0, 63, 67, 64]
    target_channel_indices = [0, 1, 2, 3] + [4 + idx for idx in target_coco_indices]

    orig_output = graph.output[0]
    orig_output_name = orig_output.name

    internal_out_name = "raw_yolo_output_84_v3"
    for node in graph.node:
        for i, out_name in enumerate(node.output):
            if out_name == orig_output_name:
                node.output[i] = internal_out_name

    indices_tensor = helper.make_tensor(
        name="gather_channel_indices_v3",
        data_type=onnx.TensorProto.INT64,
        dims=[8],
        vals=target_channel_indices,
    )
    graph.initializer.append(indices_tensor)

    gather_node = helper.make_node(
        op_type="Gather",
        inputs=[internal_out_name, "gather_channel_indices_v3"],
        outputs=["gathered_channels_v3"],
        name="gather_4class_v3",
        axis=1,
    )
    graph.node.append(gather_node)

    # Balanced sensitivity scaling for small objects in letterboxed space
    scale_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.20]
    scale_tensor = helper.make_tensor(
        name="class_calibration_scales_v3",
        data_type=onnx.TensorProto.FLOAT,
        dims=[1, 8, 1],
        vals=scale_factors,
    )
    graph.initializer.append(scale_tensor)

    mul_node = helper.make_node(
        op_type="Mul",
        inputs=["gathered_channels_v3", "class_calibration_scales_v3"],
        outputs=[orig_output_name],
        name="apply_v3_calibration",
    )
    graph.node.append(mul_node)

    graph.output[0].type.tensor_type.shape.dim[1].dim_value = 8

    # Metadata
    metadata = {
        "model_name": "YOLO-WatchGuard-Assets-v3",
        "phase": "5.19D",
        "nc": "4",
        "classes": "person,laptop,cell_phone,mouse",
        "preprocessing": "letterbox",
        "input_resolution": "640x640",
    }
    for k, v in metadata.items():
        prop = model.metadata_props.add()
        prop.key = k
        prop.value = v

    os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
    onnx.save(model, output_onnx)
    print(f"Exported V3 model to: {output_onnx} ({os.path.getsize(output_onnx)/(1024*1024):.2f} MB)")

    # Verify OpenCV DNN
    net = cv2.dnn.readNetFromONNX(output_onnx)
    blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
    net.setInput(blob)
    out = net.forward()
    print(f"OpenCV DNN verification successful. Output shape: {out.shape}")
    assert out.shape == (1, 8, 8400)

if __name__ == "__main__":
    export_v3_model()
