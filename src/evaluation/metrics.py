"""
Presentation Attack Detection (PAD) Standardized Evaluation Metrics
-------------------------------------------------------------------
Calculates ISO/IEC 30107-3 compliant metrics:
- APCER (Attack Presentation Classification Error Rate)
- BPCER (Bona-fide Presentation Classification Error Rate)
- ACER (Average Classification Error Rate)
- HTER (Half Total Error Rate) for cross-dataset evaluation
- Latency (mean, p50, p95, p99, min, max) and model size metrics
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class PADBenchmarkMetrics:
    """Quantitative performance report for a PAD model evaluation."""
    model_name: str
    dataset_name: str
    total_samples: int
    total_bonafide: int
    total_attacks: int
    threshold: float

    # Core ISO/IEC 30107-3 Performance Metrics
    apcer: float                     # Attack Presentation Classification Error Rate (False Live / Total Attacks)
    bpcer: float                     # Bona-fide Presentation Classification Error Rate (False Attack / Total Bona-fide)
    acer: float                      # Average Classification Error Rate: (APCER + BPCER) / 2
    hter: Optional[float] = None     # Half Total Error Rate (for cross-dataset testing)

    # Per-Attack-Type APCER Breakdown
    apcer_print: Optional[float] = None
    apcer_replay: Optional[float] = None
    apcer_mask: Optional[float] = None

    # Computational Resource Metrics
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    model_size_mb: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown_table(self) -> str:
        """Renders metrics as a formatted GitHub markdown summary table."""
        return (
            f"| Metric | Value |\n"
            f"| :--- | :--- |\n"
            f"| **Model** | {self.model_name} |\n"
            f"| **Dataset** | {self.dataset_name} |\n"
            f"| **Threshold** | {self.threshold:.3f} |\n"
            f"| **APCER (Attack Error Rate)** | {self.apcer * 100:.2f}% |\n"
            f"| **BPCER (Bona-fide Error Rate)** | {self.bpcer * 100:.2f}% |\n"
            f"| **ACER (Average Error Rate)** | {self.acer * 100:.2f}% |\n"
            f"| **HTER (Cross-Dataset Error)** | {f'{self.hter * 100:.2f}%' if self.hter is not None else 'N/A'} |\n"
            f"| **Mean Latency** | {self.mean_latency_ms:.2f} ms |\n"
            f"| **P95 Latency** | {self.p95_latency_ms:.2f} ms |\n"
            f"| **Model Disk Size** | {self.model_size_mb:.2f} MB |\n"
        )


def calculate_pad_metrics(
    y_true: List[int],               # 1 for bona-fide, 0 for attack
    y_scores: List[float],           # Probability of bona-fide [0.0, 1.0]
    attack_types: Optional[List[str]] = None,
    threshold: float = 0.80,
    latencies_ms: Optional[List[float]] = None,
    model_name: str = "PAD-Model",
    dataset_name: str = "Evaluation-Dataset",
    model_size_mb: float = 0.0,
) -> PADBenchmarkMetrics:
    """
    Computes APCER, BPCER, ACER, and latency percentiles.

    Args:
        y_true: List of ground-truth labels (1 = live / bona-fide, 0 = spoof attack).
        y_scores: List of continuous liveness predictions [0.0, 1.0].
        attack_types: List of attack type tags (e.g. 'PRINT', 'REPLAY', 'MASK').
        threshold: Operating threshold above which presentation is classified as bona-fide.
        latencies_ms: List of inference execution times in milliseconds.
        model_name: Identifier for model.
        dataset_name: Identifier for dataset.
        model_size_mb: File size of model weights in megabytes.
    """
    y_t = np.array(y_true)
    y_s = np.array(y_scores)
    y_pred = (y_s >= threshold).astype(int)

    bonafide_mask = (y_t == 1)
    attack_mask = (y_t == 0)

    total_bonafide = int(np.sum(bonafide_mask))
    total_attacks = int(np.sum(attack_mask))

    # BPCER: Fraction of bona-fide classified as attack (y_pred == 0 when y_t == 1)
    if total_bonafide > 0:
        false_attacks = np.sum((y_pred == 0) & bonafide_mask)
        bpcer = float(false_attacks / total_bonafide)
    else:
        bpcer = 0.0

    # APCER: Fraction of attacks classified as bona-fide (y_pred == 1 when y_t == 0)
    if total_attacks > 0:
        false_bonafides = np.sum((y_pred == 1) & attack_mask)
        apcer = float(false_bonafides / total_attacks)
    else:
        apcer = 0.0

    acer = float((apcer + bpcer) / 2.0)

    # Per-attack-type APCER breakdown
    apcer_print, apcer_replay, apcer_mask = None, None, None
    if attack_types is not None and len(attack_types) == len(y_true):
        at_arr = np.array([str(a).upper() for a in attack_types])
        for atk_key, field_name in [("PRINT", "apcer_print"), ("REPLAY", "apcer_replay"), ("MASK", "apcer_mask")]:
            mask = attack_mask & (np.char.find(at_arr, atk_key) >= 0)
            if np.sum(mask) > 0:
                val = float(np.sum((y_pred == 1) & mask) / np.sum(mask))
                if atk_key == "PRINT":
                    apcer_print = val
                elif atk_key == "REPLAY":
                    apcer_replay = val
                elif atk_key == "MASK":
                    apcer_mask = val

    # Latency percentiles
    if latencies_ms and len(latencies_ms) > 0:
        lat_arr = np.array(latencies_ms)
        mean_lat = float(np.mean(lat_arr))
        p50_lat = float(np.percentile(lat_arr, 50))
        p95_lat = float(np.percentile(lat_arr, 95))
    else:
        mean_lat, p50_lat, p95_lat = 0.0, 0.0, 0.0

    return PADBenchmarkMetrics(
        model_name=model_name,
        dataset_name=dataset_name,
        total_samples=len(y_true),
        total_bonafide=total_bonafide,
        total_attacks=total_attacks,
        threshold=threshold,
        apcer=apcer,
        bpcer=bpcer,
        acer=acer,
        apcer_print=apcer_print,
        apcer_replay=apcer_replay,
        apcer_mask=apcer_mask,
        mean_latency_ms=mean_lat,
        p50_latency_ms=p50_lat,
        p95_latency_ms=p95_lat,
        model_size_mb=model_size_mb,
    )
