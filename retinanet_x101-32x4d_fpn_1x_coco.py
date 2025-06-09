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