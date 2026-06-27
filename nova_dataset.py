from typing import Any, Iterator
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

NOVA_DATASET = "hf://datasets/c-i-ber/Nova/data/nova-v1.parquet"
NOVA_REPO = "c-i-ber/Nova"

class NovaDataset:
    def __init__(self):
        self.ds = load_dataset("parquet", data_files=NOVA_DATASET, split="train")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.ds)  # type: ignore[arg-type]

    def load_image(self, image_path: str) -> Image.Image:
        local_path = hf_hub_download(repo_id=NOVA_REPO, filename=image_path, repo_type="dataset")
        return Image.open(local_path)
