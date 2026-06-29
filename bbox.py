from dataclasses import dataclass
from abc import ABC, abstractmethod
from PIL import Image

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    confidence: float
    label: str = ""

    @staticmethod
    def from_xywh(x: float, y: float, width: float, height: float, confidence: float, label: str = "") -> "BBox":
        return BBox(x1=x, y1=y, x2=x + width, y2=y + height, width=width, height=height, confidence=confidence, label=label)


class BBoxParser(ABC):
    @abstractmethod
    def localise_anomalies(self, image: Image.Image, prompt: str) -> list[BBox]:
        ...
