from bbox import BBoxParser, BBox
from nova_dataset import NovaDataset
from torch import tensor, zeros
from torchmetrics.detection import MeanAveragePrecision

class Evaluation:
    def __init__(self, dataset: NovaDataset):
        self.dataset = dataset

    def eval_anomaly_localisation(self, bbox_preds: list[BBox]) -> MeanAveragePrecision:
        gt = []
        preds = []
        
        for row in self.dataset:
            # image = self.dataset.load_image(row["image_path"])
            gt_bboxes = row["bboxes"]
            gt_boxes = tensor([
                [b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]]
                for b in gt_bboxes
            ]) if gt_bboxes else zeros((0, 4))
            gt.append(dict(
                boxes=gt_boxes,
                labels=tensor([0] * len(gt_bboxes)),
            ))

            pred_boxes = tensor([
                [b.x1, b.y1, b.x2, b.y2] for b in bbox_preds
            ]) if bbox_preds else zeros((0, 4))
            pred_scores = tensor([b.confidence for b in bbox_preds])
            preds.append(dict(
                boxes=pred_boxes,
                scores=pred_scores,
                labels=tensor([0] * len(bbox_preds)),
            ))

        map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        map.update(preds, gt)
        return map
