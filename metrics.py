import os
import json
from pathlib import Path


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    :param box1: List [x_left, y_top, x_right, y_bottom]
    :param box2: List [x_left, y_top, x_right, y_bottom]
    :return: IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def convert_coco_to_box(coco_bbox):
    """
    Convert COCO bounding box format [x_min, y_min, width, height]
    to box format [x_left, y_top, x_right, y_bottom].
    """
    x_min, y_min, width, height = coco_bbox
    x_right = x_min + width
    y_bottom = y_min + height
    return [x_min, y_min, x_right, y_bottom]


def calculate_metrics(ground_truths, predictions, iou_threshold):
    """
    Calculate precision, recall, and F1-score for a single file.
    :param ground_truths: List of ground truth bounding boxes.
    :param predictions: List of predicted bounding boxes.
    :param iou_threshold: IoU threshold to consider a prediction correct.
    :return: Precision, recall, F1-score, true positives, false positives, false negatives.
    """
    true_positives = 0 # dog in the image, model sees the dog
    false_positives = 0 # dog is not in the image, model sees the dog
    false_negatives = 0 # dog is in the image, model does not see the dog
    total_iou = 0

    ground_truths_not_predicted = ground_truths.copy()
    gt_matched = []
    for gt_bbox in ground_truths:
        for pred_idx, pred_bbox in enumerate(predictions):
            iou = calculate_iou(pred_bbox, gt_bbox)
            total_iou += iou

            if iou >= iou_threshold and pred_idx not in gt_matched:
                true_positives += 1
                gt_matched.append(pred_idx)
                if gt_bbox in ground_truths_not_predicted: ground_truths_not_predicted.remove(gt_bbox)
            else:
                false_positives += 1

    false_negatives = len(ground_truths_not_predicted)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    average_iou = total_iou / len(predictions) if len(predictions) != 0 else 0

    return precision, recall, f1_score, true_positives, false_positives, false_negatives, average_iou


def main():
    # Update these paths
    ground_truth_file = Path(
        r"C:\Users\User\Desktop\Wound_Detection\MMDet_conda38_V2\dataset\test\fold_1\fold_1.json"
    )
    prediction_dir = Path(
        r"C:\Users\User\Desktop\Wound_Detection\MMDet_conda38_V2\work_dirs\retinanet_x101-32x4d_fpn_1x_coco\FolderCreatedWhenTestingTheModel\output\preds"
    )
    output_file = Path(
        r"C:\Users\User\Desktop\Wound_Detection\MMDet_conda38_V2\work_dirs\retinanet_x101-32x4d_fpn_1x_coco\FolderCreatedWhenTestingTheModel\output\preds\results.json")
    iou_threshold = 0.5  # Set IoU threshold

    # Load ground truths
    with open(ground_truth_file, "r") as f:
        gt_data = json.load(f)

    gt_image_mapping = {Path(image["file_name"]).stem: image["id"] for image in gt_data["images"]}
    gt_annotations = {}
    for ann in gt_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in gt_annotations:
            gt_annotations[image_id] = []
        gt_annotations[image_id].append(convert_coco_to_box(ann["bbox"]))

    results = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_average_iou = 0
    how_many = 0

    for pred_file in prediction_dir.glob("*.json"):
        file_name = pred_file.stem
        if file_name not in gt_image_mapping:
            print(f"Warning: Ground truth for {file_name} not found. Skipping.")
            continue

        image_id = gt_image_mapping[file_name]
        ground_truth_bboxes = gt_annotations.get(image_id, [])

        with open(pred_file, "r") as f:
            pred_data = json.load(f)

        pred_bboxes = pred_data["bboxes"]  # Already in [x_left, y_top, x_right, y_bottom] format

        precision, recall, f1_score, tp, fp, fn, average_iou = calculate_metrics(ground_truth_bboxes, pred_bboxes, iou_threshold)
        results[file_name] = {"precision": precision, "recall": recall, "f1_score": f1_score, "average_iou": average_iou}

        # Update cumulative metrics
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_precision += precision
        total_recall += recall
        total_f1_score += f1_score
        total_average_iou += average_iou
        how_many += 1

    # Calculate overall metrics
    overall_average_precision = total_precision / how_many if how_many > 0 else 0
    overall_average_recall = total_recall / how_many if how_many > 0 else 0
    overall_average_f1_score = total_f1_score / how_many if how_many > 0 else 0
    overall_average_iou = total_average_iou / how_many if how_many > 0 else 0

    results["overall"] = {
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": overall_average_precision,
        "recall": overall_average_recall,
        "f1_score": overall_average_f1_score,
        "average_iou": overall_average_iou,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
