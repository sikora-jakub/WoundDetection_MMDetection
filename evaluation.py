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