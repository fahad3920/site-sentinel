"""Site Sentinel - construction safety & progress tracking (CPU-only).

PPE detection uses the Apache-2.0 YOLO11 safety model
(wesjos/Yolo-hard-hat-safety-vest) so the project stays commercially usable
without the Ultralytics AGPL obligation.
"""
from . import config

__all__ = ["config"]
__version__ = "0.1.0"
