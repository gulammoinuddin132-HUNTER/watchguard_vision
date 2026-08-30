"""
WatchGuard Vision - Unified 84-Class Training & Validation Pipeline (Phase 5.21)
---------------------------------------------------------------------------------
Prepares training hyperparameters, model architecture selection (YOLOv8n / YOLO11n),
and multi-scale small-object loss optimization for the 84-class WatchGuard detector.

Safety Guard:
  Strictly enforces pre-training dataset audit. Will NOT train on empty or synthetic mock data.
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.audit_unified_dataset import UnifiedDatasetAuditor

logger = logging.getLogger("WatchGuardVision.TrainUnified84")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class UnifiedTrainingConfig:
    """Hyperparameter and architectural configuration for 84-class unified model training."""
    model_family: str = "yolo11n.pt"  # Options: "yolov8n.pt", "yolo11n.pt"
    dataset_yaml: Path = REPO_ROOT / "data" / "datasets" / "watchguard_unified_84" / "watchguard_unified_84.yaml"
    imgsz: int = 640
    batch_size: int = 16
    epochs: int = 100
    lr0: float = 0.01
    lrf: float = 0.01
    mosaic: float = 1.0
    copy_paste: float = 0.3
    fl_gamma: float = 2.0  # Focal loss gamma for rare/small object focus
    seed: int = 42
    workers: int = 4
    device: str = "cpu"  # "0" for GPU if available, else "cpu"
    min_required_images: int = 100  # Safety threshold to prevent training on empty datasets


class Unified84Trainer:
    """Orchestrates model training, export to ONNX, and COCO retention auditing."""

    def __init__(self, config: Optional[UnifiedTrainingConfig] = None):
        self.config = config or UnifiedTrainingConfig()
        self.dataset_dir = self.config.dataset_yaml.parent
        self.auditor = UnifiedDatasetAuditor(self.dataset_dir)

    def prepare_and_validate_dataset(self) -> bool:
        """Audits dataset before training to ensure zero data leakage and valid labels."""
        logger.info(f"Auditing dataset at: {self.dataset_dir}")
        report = self.auditor.run_full_audit()

        if report["status"] != "PASSED":
            logger.error("Dataset audit failed. Aborting training.")
            for err in report["errors"]:
                logger.error(f"  - {err}")
            return False

        if report["total_images"] < self.config.min_required_images:
            logger.warning(
                f"Dataset contains only {report['total_images']} images (minimum required: {self.config.min_required_images}). "
                f"Training pipeline is initialized in FOUNDATION PREPARATION mode. Awaiting real training image ingestion."
            )
            return False

        logger.info(f"Dataset audit passed with {report['total_images']} images and {report['total_annotations']} annotations.")
        return True

    def train_model(self) -> Optional[Path]:
        """
        Executes YOLO training session using Ultralytics backend.
        Refuses to run if dataset has insufficient real data.
        """
        if not self.prepare_and_validate_dataset():
            logger.info("Training halted: dataset requires image collection phase before training can commence.")
            return None

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            logger.error("Ultralytics package not installed. Run 'pip install ultralytics'.")
            return None

        logger.info(f"Initiating training with backbone: {self.config.model_family} (84 classes)")
        model = YOLO(self.config.model_family)
        results = model.train(
            data=str(self.config.dataset_yaml),
            imgsz=self.config.imgsz,
            batch=self.config.batch_size,
            epochs=self.config.epochs,
            lr0=self.config.lr0,
            lrf=self.config.lrf,
            mosaic=self.config.mosaic,
            copy_paste=self.config.copy_paste,
            fl_gamma=self.config.fl_gamma,
            seed=self.config.seed,
            device=self.config.device,
            workers=self.config.workers,
            plots=True,
        )
        logger.info("Training session completed. Exporting model to ONNX...")
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        best_model = YOLO(str(best_pt))
        exported_onnx = best_model.export(format="onnx", imgsz=self.config.imgsz, dynamic=False)
        return Path(exported_onnx)


def main():
    trainer = Unified84Trainer()
    trainer.prepare_and_validate_dataset()


if __name__ == "__main__":
    main()
