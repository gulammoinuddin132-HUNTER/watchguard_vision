"""
WatchGuard Vision - Research Evaluation Module
---------------------------------------------
Standalone evaluation harness and metrics for presentation attack detection.
"""

from src.evaluation.metrics import PADBenchmarkMetrics, calculate_pad_metrics
from src.evaluation.pad_harness import DatasetSample, PADBenchmarkHarness

__all__ = [
    "DatasetSample",
    "PADBenchmarkMetrics",
    "calculate_pad_metrics",
    "PADBenchmarkHarness",
]
