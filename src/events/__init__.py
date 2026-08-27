"""
Events Package for WatchGuard Vision
"""
from src.events.event_types import EventType, RiskLevel
from src.events.event_logger import EventLogger

__all__ = ["EventType", "RiskLevel", "EventLogger"]
