from bbox import BBoxParser, BBox
from nova_dataset import NovaDataset


class Evaluation:
    _LOCALISATION_PROMPT = "Detect and localise all abnormalities in this brain MRI scan."

    @classmethod
    def eval_anomaly_localisation(cls, dataset: NovaDataset, model: BBoxParser):
        for row in dataset:
            image = dataset.load_image(row["image_path"])
            gt_bboxes = [
                BBox(x=b["x"], y=b["y"], width=b["width"], height=b["height"], label=b["label"])
                for b in row["bboxes"]
            ]
            pred_bboxes = model.localise_anomalies(image, Evaluation._LOCALISATION_PROMPT)
            