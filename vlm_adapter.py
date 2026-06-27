from abc import ABC, abstractmethod
from PIL import Image


class VLMAdapter(ABC):
    @abstractmethod
    def describe_image(self, image: Image.Image, prompt: str) -> str:
        ...

    @abstractmethod
    def diagnose(self, image: Image.Image, clinical_history: str, caption_text: str, prompt: str) -> str:
        ...
