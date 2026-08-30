"""
WatchGuard Vision - Research Evaluation Metrics Engine (Part 13)
----------------------------------------------------------------
Computes standard biometric and contextual security metrics:
  - FAR (False Acceptance Rate)
  - FRR (False Rejection Rate)
  - FAUS (False Access Under Spoof)
  - FAPC (False Access Under Prohibited Context)
  - Overall False Access Rate
  - Decision Latency & Throughput
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EvaluationMetrics:
    """Quantitative security and operational performance metrics."""
    total_evaluations: int = 0
    genuine_attempts: int = 0
    impostor_attempts: int = 0
    spoof_attempts: int = 0
    prohibited_context_attempts: int = 0

    false_accepts: int = 0
    false_rejects: int = 0
    spoof_accepts: int = 0
    prohibited_context_accepts: int = 0

    total_latency_ms: float = 0.0
    evaluation_times: List[float] = field(default_factory=list)

    @property
    def far(self) -> float:
        """False Acceptance Rate (FAR) in [0.0, 1.0]."""
        if self.impostor_attempts == 0:
            return 0.0
        return round(self.false_accepts / self.impostor_attempts, 4)

    @property
    def frr(self) -> float:
        """False Rejection Rate (FRR) in [0.0, 1.0]."""
        if self.genuine_attempts == 0:
            return 0.0
        return round(self.false_rejects / self.genuine_attempts, 4)

    @property
    def faus(self) -> float:
        """False Access Under Spoof (FAUS) in [0.0, 1.0]."""
        if self.spoof_attempts == 0:
            return 0.0
        return round(self.spoof_accepts / self.spoof_attempts, 4)

    @property
    def fapc(self) -> float:
        """False Access Under Prohibited Context (FAPC) in [0.0, 1.0]."""
        if self.prohibited_context_attempts == 0:
            return 0.0
        return round(self.prohibited_context_accepts / self.prohibited_context_attempts, 4)

    @property
    def false_access_rate(self) -> float:
        """Overall False Access Rate across all unauthorized/spoof/prohibited attempts."""
        total_threats = self.impostor_attempts + self.spoof_attempts + self.prohibited_context_attempts
        if total_threats == 0:
            return 0.0
        total_breaches = self.false_accepts + self.spoof_accepts + self.prohibited_context_accepts
        return round(total_breaches / total_threats, 4)

    @property
    def avg_latency_ms(self) -> float:
        """Average decision latency in milliseconds."""
        if self.total_evaluations == 0:
            return 0.0
        return round(self.total_latency_ms / self.total_evaluations, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluations": self.total_evaluations,
            "genuine_attempts": self.genuine_attempts,
            "impostor_attempts": self.impostor_attempts,
            "spoof_attempts": self.spoof_attempts,
            "prohibited_context_attempts": self.prohibited_context_attempts,
            "far": self.far,
            "frr": self.frr,
            "faus": self.faus,
            "fapc": self.fapc,
            "false_access_rate": self.false_access_rate,
            "avg_latency_ms": self.avg_latency_ms,
        }


def format_metrics_table(results_by_model: Dict[str, EvaluationMetrics]) -> str:
    """Renders a formatted Markdown table comparing baselines to WatchGuard CCSDF."""
    lines = [
        "| Architecture / Model | FAR (%) | FRR (%) | FAUS (Spoof Acc %) | FAPC (Context Acc %) | False Access Rate (%) | Avg Latency (ms) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for name, m in results_by_model.items():
        far_pct = f"{m.far * 100:.1f}%"
        frr_pct = f"{m.frr * 100:.1f}%"
        faus_pct = f"{m.faus * 100:.1f}%"
        fapc_pct = f"{m.fapc * 100:.1f}%"
        overall_pct = f"{m.false_access_rate * 100:.1f}%"
        lat_ms = f"{m.avg_latency_ms:.3f} ms"
        lines.append(f"| **{name}** | {far_pct} | {frr_pct} | {faus_pct} | {fapc_pct} | {overall_pct} | {lat_ms} |")
    return "\n".join(lines)
