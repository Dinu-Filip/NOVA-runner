import argparse
import importlib

from evaluation import Evaluation
from nova_dataset import NovaDataset
from vlm_adapter import VLMAdapter


def load_adapter(spec: str) -> VLMAdapter:
    try:
        module_path, _, class_name = spec.partition(":")
        if not module_path or not class_name:
            raise ValueError(
                f"Adapter spec must be 'module.path:ClassName', got {spec!r}"
            )
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise SystemExit(f"Could not load adapter {spec!r}: {e}")

    if not (isinstance(cls, type) and issubclass(cls, VLMAdapter)):
        raise SystemExit(f"{spec!r} is not a VLMAdapter subclass")

    return cls()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NOVA benchmark against a VLM.")
    parser.add_argument(
        "adapter",
        help="VLM adapter to evaluate, as 'module.path:ClassName' "
        "(e.g. 'adapters.my_vlm:MyVLM')",
    )
    args = parser.parse_args()

    adapter = load_adapter(args.adapter)

    dataset = NovaDataset()
    evaluation = Evaluation(dataset)

    print(f"Loaded adapter: {type(adapter).__name__}")
    print(dataset.ds.features)
