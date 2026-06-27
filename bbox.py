from dataclasses import dataclass
from abc import ABC, abstractmethod
from PIL import Image

@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float
    label: str = ""


class BBoxParser(ABC):
    @abstractmethod
    def localise_anomalies(self, image: Image.Image, prompt: str) -> list[BBox]:
        ...
