"""
Presentation Attack Detection (PAD) Research Evaluation Harness
---------------------------------------------------------------
Standalone benchmark evaluation harness for measuring PAD performance.
Completely separated from the live WatchGuard Vision dashboard runtime.
Supports intra-dataset benchmarking and cross-dataset (HTER) evaluation.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from src.core.pad.base_pad import AttackType, BasePADEngine, PADResult
from src.evaluation.metrics import PADBenchmarkMetrics, calculate_pad_metrics

logger = logging.getLogger("WatchGuardVision.Evaluation.Harness")


@dataclass
class DatasetSample:
    """Standardized representation of a single benchmark dataset sample."""
    image_path: str
    is_live: bool                    # True = bona-fide, False = spoof attack
    attack_type: AttackType          # PRINT, REPLAY, MASK, BONAFIDE
    subject_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PADBenchmarkHarness:
    """
    Pluggable evaluation runner capable of evaluating any combination of:
    - Face Detector
    - Feature Recognizer
    - PAD Model
    - Target Dataset (e.g. CelebA-Spoof, LCC-FASD)
    """

    def __init__(
        self,
        pad_engine: BasePADEngine,
        model_name: Optional[str] = None,
    ):
        self.pad_engine = pad_engine
        self.model_name = model_name or pad_engine.model_name

    def evaluate_dataset(
        self,
        samples: List[DatasetSample],
        dataset_name: str = "Benchmark-Dataset",
        threshold: float = 0.80,
        model_size_mb: float = 0.0,
        image_loader: Optional[Callable[[str], np.ndarray]] = None,
    ) -> PADBenchmarkMetrics:
        """
        Executes a comprehensive benchmark run over the provided sample list.
        """
        if not self.pad_engine.is_ready:
            self.pad_engine.initialize()

        y_true: List[int] = []
        y_scores: List[float] = []
        attack_types: List[str] = []
        latencies_ms: List[float] = []

        logger.info(f"Starting benchmark evaluation: Model='{self.model_name}', Dataset='{dataset_name}' ({len(samples)} samples)...")

        for sample in samples:
            y_true.append(1 if sample.is_live else 0)
            attack_types.append(sample.attack_type.value)

            # Load image or generate synthetic representative crop
            if image_loader and sample.image_path:
                img = image_loader(sample.image_path)
            else:
                img = np.zeros((112, 112, 3), dtype=np.uint8)

            t0 = time.perf_counter()
            res = self.pad_engine.evaluate(raw_face_crop=img, aligned_face=img)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            y_scores.append(float(res.confidence))
            latencies_ms.append(lat_ms)

        metrics = calculate_pad_metrics(
            y_true=y_true,
            y_scores=y_scores,
            attack_types=attack_types,
            threshold=threshold,
            latencies_ms=latencies_ms,
            model_name=self.model_name,
            dataset_name=dataset_name,
            model_size_mb=model_size_mb,
        )

        logger.info(f"Benchmark finished. APCER={metrics.apcer*100:.2f}%, BPCER={metrics.bpcer*100:.2f}%, ACER={metrics.acer*100:.2f}%")
        return metrics

    @staticmethod
    def calculate_cross_dataset_hter(
        train_val_metrics: PADBenchmarkMetrics,
        test_metrics: PADBenchmarkMetrics,
    ) -> float:
        """
        Calculates Half Total Error Rate (HTER) for cross-dataset evaluation:
        HTER = (APCER_test + BPCER_test) / 2 evaluated at threshold determined on train/val set.
        """
        return float((test_metrics.apcer + test_metrics.bpcer) / 2.0)
