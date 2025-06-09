```python
# Install PyTorch along with torchvision and torchaudio, specifying the version compatible with your CUDA for GPU acceleration (in our instance it is CUDA 12.1)
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


```python
# For full installation guide please head to the MMDetection documentation online.

# Install or update OpenMIM, a tool for managing and installing OpenMMLab packages.
!pip install -U openmim
```


```python
# Install MMEngine, the foundational library for OpenMMLab projects, which provides essential functionalities like configuration and logging.
!mim install mmengine
```


```python
# Install MMCV (MMCV-Core and MMCV-Full), which provides core utilities and modules for computer vision tasks. A specific version range is used for compatibility.
!mim install "mmcv>=2.0.0rc4, <2.2.0"
```


```python
# Clone the MMDetection repository, which contains the framework and tools for object detection tasks.
!git clone https://github.com/open-mmlab/mmdetection.git
```


```python
# Install the MMDetection package using OpenMIM, which enables quick setup and ensures dependency management.
!mim install mmdet
```


```python
# After installing MMDetection, configure the RetinaNet model by modifying the configuration file:

# The new config inherits a base config to highlight the necessary modifications
_base_ = r'C:\Users\User\Desktop\Wound_Detection\MMDet_conda38_V2\mmdetection\configs\retinanet\retinanet_x101-32x4d_fpn_1x_coco.py'

# Update model parameters
model = dict(
    bbox_head=dict(num_classes=1)  # Set number of classes to 1 (for wound detection)
)

# Define dataset settings
data_root = r'C:\Users\User\Desktop\Wound_Detection\MMDet_conda38_V2\dataset'
metainfo = {
    'classes': 'wound'
}

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=r'train\f3-f10\f3-f10.json',   # change this when switching to the new fold
        data_prefix=dict(img=r'train\f3-f10')   # change this when switching to the new fold
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=r'val\fold_2\fold_2.json',     # change this when switching to the new fold
        data_prefix=dict(img=r'val\fold_2')     # change this when switching to the new fold
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=r'test\fold_1\fold_1.json',    # change this when switching to the new fold
        data_prefix=dict(img=r'test\fold_1')    # change this when switching to the new fold
    )
)

# Update evaluators
val_evaluator = dict(ann_file=data_root + r'val\fold_2\fold_2.json')    # change this when switching to the new fold
test_evaluator = dict(ann_file=data_root + r'test\fold_1\fold_1.json')  # change this when switching to the new fold

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,  # Train for 300 epochs
    val_interval=1   # Validate after every epoch
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_best='auto'),
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=75,
        min_delta=0.005
    )
)

# Load a pre-trained RetinaNet model
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth'
```


```python
# Place the dataset in the dataset folder. In each folder there has to be a single json file describing the ground truth bounding boxes with the following COCO structure (e.g. fold_1.json):
{
    "images": [
        {
            "id": 1,
            "file_name": "case_1_1_1_rgb.png",
            "width": 320,
            "height": 240
        },
        {
            "id": 2,
            "file_name": "case_26_1_1_rgb.png",
            "width": 320,
            "height": 240
        },
        {
            "id": 3,
            "file_name": "case_26_7_1_rgb.png",
            "width": 320,
            "height": 240
        },
        {
            "id": 4,
            "file_name": "case_42_1_1_rgb.png",
            "width": 320,
            "height": 240
        },
        {
            "id": 5,
            "file_name": "case_42_7_1_rgb.png",
            "width": 320,
            "height": 240
        },
        {
            "id": 6,
            "file_name": "case_47_1_1_rgb.png",
            "width": 320,
            "height": 240
        },
        {
            "id": 7,
            "file_name": "case_47_70_1_rgb.png",
            "width": 320,
            "height": 240
        },
        {
            "id": 8,
            "file_name": "case_6_1_1_rgb.png",
            "width": 320,
            "height": 240
        },
        {
            "id": 9,
            "file_name": "case_7_1_2_rgb.png",
            "width": 320,
            "height": 240
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [
                218.85199999999998,
                115.74048,
                22.22816,
                20.47632
            ],
            "area": 455.1509171712,
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": 2,
            "category_id": 0,
            "bbox": [
                162.73376000000002,
                109.71816,
                12.428159999999998,
                11.6616
            ],
            "area": 144.93223065599997,
            "iscrowd": 0
        },
        {
            "id": 3,
            "image_id": 3,
            "category_id": 0,
            "bbox": [
                138.86303999999998,
                127.73064,
                12.20928,
                12.482880000000002
            ],
            "area": 152.40697712640002,
            "iscrowd": 0
        },
        {
            "id": 4,
            "image_id": 4,
            "category_id": 0,
            "bbox": [
                105.83888,
                66.66684,
                56.1488,
                82.56216
            ],
            "area": 4635.766209408001,
            "iscrowd": 0
        },
        {
            "id": 5,
            "image_id": 5,
            "category_id": 0,
            "bbox": [
                95.8328,
                111.65208,
                68.32736,
                39.67392
            ],
            "area": 2710.8142144512003,
            "iscrowd": 0
        },
        {
            "id": 6,
            "image_id": 6,
            "category_id": 0,
            "bbox": [
                112.74768,
                108.84215999999999,
                56.50144,
                31.31664
            ],
            "area": 1769.4352559616,
            "iscrowd": 0
        },
        {
            "id": 7,
            "image_id": 7,
            "category_id": 0,
            "bbox": [
                146.66112,
                122.71307999999999,
                18.46336,
                19.59384
            ],
            "area": 361.76812170240004,
            "iscrowd": 0
        },
        {
            "id": 8,
            "image_id": 7,
            "category_id": 0,
            "bbox": [
                164.87327999999997,
                113.66976,
                24.36672,
                24.303839999999997
            ],
            "area": 592.2048642048,
            "iscrowd": 0
        },
        {
            "id": 9,
            "image_id": 7,
            "category_id": 0,
            "bbox": [
                151.81072,
                135.83844000000002,
                44.46303999999999,
                45.216719999999995
            ],
            "area": 2010.4728300287995,
            "iscrowd": 0
        },
        {
            "id": 10,
            "image_id": 8,
            "category_id": 0,
            "bbox": [
                120.67008000000001,
                89.50164,
                64.59072,
                94.92888
            ],
            "area": 6131.524707993601,
            "iscrowd": 0
        },
        {
            "id": 11,
            "image_id": 9,
            "category_id": 0,
            "bbox": [
                161.47472,
                66.68496,
                22.8304,
                27.64848
            ],
            "area": 631.225857792,
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "wound",
            "supercategory": "none"
        }
    ]
}
```


```python
# Create the evaluation.py script to return bounding box predictions:

#Basic Usage
from mmdet.apis import DetInferencer

# Choose to use a config
config_path = r'C:\Users\User\Desktop\Wound_Detection\MMDet_conda38_V2\retinanet_x101-32x4d_fpn_1x_coco.py'

# Setup a checkpoint file to load
checkpoint = r'work_dirs\retinanet_x101-32x4d_fpn_1x_coco\FolderCreatedWhenTrainingTheModel\best_coco_bbox_mAP_epoch_X.pth'

# Setup out directory
output = r'work_dirs\retinanet_x101-32x4d_fpn_1x_coco\FolderCreatedWhenTestingTheModel\output'

# Setup the fold
selected_fold = r'C:\Users\User\Desktop\Wound_Detection\MMDet_conda38_V2\dataset\test\fold_1'

# Initialize the DetInferencer
inferencer = DetInferencer(model=config_path, weights=checkpoint, device='cuda:0', show_progress=True)

# Perform inference
inferencer(inputs=selected_fold, batch_size=1, print_result=True, show=False, out_dir=output, no_save_vis=False, no_save_pred=False)
```


```python
# Create the metrics.py script to compute additional metrics:

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

```


```python
# RetinaNet training
# Train the RetinaNet model using the specified configuration file. The config file contains hyperparameters, model architecture, dataset details, and more.
# Remember that you have to set the correct train, val, and test folds in the config file!
python mmdetection/tools/train.py PathToTheConfigFile\retinanet_x101-32x4d_fpn_1x_coco.py

# RetinaNet testing
# Test the trained RetinaNet model using the specified configuration file and the best checkpoint file. The results will be saved in the 'output' directory.
python mmdetection/tools/test.py PathToTheConfigFile\retinanet_x101-32x4d_fpn_1x_coco.py work_dirs\retinanet_x101-32x4d_fpn_1x_coco\best_coco_bbox_mAP_epoch_X.pth --show-dir output

# Now remember to put the best_coco_bbox_mAP_epoch_X.pth into the FolderCreatedWhenTestingTheModel for safekeeping and further use!

# Run the evaluation.py script to return bounding box predictions
# Remember that you have to modify the folders' paths accordingly!
python YourPath\evaluation.py

# Run the metrics.py script to evaluate the model's performance on additional metrics
# Remember that you have to modify the folders' paths accordingly!
python YourPath\metrics.py
```

Here are the outputs returned by this specific implementation:

![title](images_for_jupyter/case_1_1_1_rgb.png)
![title](images_for_jupyter/case_6_1_1_rgb.png)
![title](images_for_jupyter/case_7_1_2_rgb.png)
![title](images_for_jupyter/case_26_1_1_rgb.png)
![title](images_for_jupyter/case_26_7_1_rgb.png)
![title](images_for_jupyter/case_42_1_1_rgb.png)
![title](images_for_jupyter/case_42_7_1_rgb.png)
![title](images_for_jupyter/case_47_1_1_rgb.png)
![title](images_for_jupyter/case_47_70_1_rgb.png)
