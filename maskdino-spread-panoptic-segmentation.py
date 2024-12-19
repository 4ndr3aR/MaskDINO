#!/usr/bin/env python3

import sys, os, json, time, random

#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

import numpy as np 
import cv2
import torch

import torch.nn.functional as F

from pathlib import Path

# Setup detectron2 logger
import detectron2
#from detectron2.utils.logger import setup_logger
#setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.logger import setup_logger

import maskdino
from maskdino.config import add_maskdino_config

root              = Path('/mnt/raid1/repos/maskdino/')
output_dir        = 'maskdino-ade20k-output'
pretrained_models = 'pretrained-models'
img_path          = [
			'images/camvid_old/images/0016E5_08141.png',
			'images/black-car-over-black-background.jpg',
			'images/car-with-complex-background.jpg',
			'images/car-with-white-background.jpg'
		]


def get_keypoints_cfg():
	# Inference with a keypoint detection model
	cfg_fn = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
	cfg_keypoint = get_cfg()   # get a fresh new config
	cfg_keypoint.merge_from_file(model_zoo.get_config_file   (cfg_fn))
	cfg_keypoint.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_fn)
	cfg_keypoint.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
	return cfg_keypoint

def get_instance_segmentation_cfg():
	# Inference with instance segmentation
	cfg_fn = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
	cfg_inst = get_cfg()
	cfg_inst.merge_from_file(model_zoo.get_config_file   (cfg_fn))
	cfg_inst.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_fn)
	cfg_inst.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	return cfg_inst

def get_panoptic_segmentation_cfg():
	# Inference with a panoptic segmentation model
	cfg_fn = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
	cfg_pan = get_cfg()
	cfg_pan.merge_from_file(model_zoo.get_config_file   (cfg_fn))
	cfg_pan.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_fn)
	return cfg_pan

def detectron_predict(img, cfg, task='keypoints', show_images=True, window_name='prediction'):
	# Find a model from detectron2's model zoo.  https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
	predictor = DefaultPredictor(cfg)
	outputs   = predictor(img)
	if task == 'keypoints' or task == 'instance segmentation':
		instances = outputs["instances"]
	elif task == 'panoptic segmentation':
		panoptic_seg, segments_info = outputs["panoptic_seg"]
	if show_images:
		v = Visualizer(img[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
		if task == 'keypoints' or task == 'instance segmentation':
			out = v.draw_instance_predictions(instances.to("cpu"))
		elif task == 'panoptic segmentation':
			out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
		cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
		cv2.imshow(window_name, out.get_image()[:, :, ::-1])
		cv2.waitKey(10000)
		cv2.destroyAllWindows()
	return outputs



from PIL import Image
from tqdm import tqdm
import pandas as pd

setup_logger()

# --- Configuration & Hyperparameters ---
NUM_CLASSES = 2  # Trees and Non-trees
MAX_ITER = 1000  # Adjust as needed
BASE_LR = 0.001
BATCH_SIZE = 2

# --- Data Loading Functions ---

def rgb_to_id(color, color_palette):
    """
    Maps an RGB color to a class ID based on a color palette.

    Args:
        color: A tuple representing the RGB color (R, G, B).
        color_palette: A list of dictionaries in the form [{0: [55, 181, 57]}, {1: [153, 108, 6]}, ...]

    Returns:
        The class ID if the color is found in the palette, otherwise None.
    """
    for entry in color_palette:
        for class_id, rgb in entry.items():
            if list(color) == rgb:  # Convert color to list for comparison
                return class_id
    return None

def get_tree_dicts(img_dir, mask_dir, color_palette):
    """
    Loads image and mask data, creates Detectron2 dataset dictionaries.

    Args:
        img_dir: Directory containing RGB images.
        mask_dir: Directory containing RGB panoptic segmentation masks.
        color_palette: A list of dictionaries defining the color mapping.

    Returns:
        A list of dataset dictionaries in Detectron2 format.
    """
    dataset_dicts = []
    idx = 0
    for img_name in tqdm(os.listdir(img_dir)):
        record = {}
        
        filename = os.path.join(img_dir, img_name)
        mask_filename = os.path.join(mask_dir, img_name)
        
        if not os.path.exists(mask_filename):
          print(f"Warning: Mask not found for {filename}. Skipping.")
          continue

        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        mask = np.array(Image.open(mask_filename))
        
        if len(mask.shape) != 3:
            print(f"Skipping image {img_name} because its mask is not RGB.")
            continue

        # Extract instance-level masks and bounding boxes
        panoptic_seg = np.zeros((height, width), dtype=np.int32)
        segments_info = []
        instance_id = 1 # instance id = 0 is reserved for background
        instance_color_mapping = {}

        for y in range(height):
            for x in range(width):
                color = tuple(mask[y, x])
                class_id = rgb_to_id(color, color_palette)

                if class_id is not None:
                    if class_id not in instance_color_mapping:
                        instance_color_mapping[class_id] = {}

                    color_str = str(color) # Use color string as a unique identifier for each instance within a class
                    if color_str not in instance_color_mapping[class_id]:
                        instance_color_mapping[class_id][color_str] = instance_id
                        instance_id +=1

                    panoptic_seg[y,x] = instance_color_mapping[class_id][color_str]

                    is_crowd = 0 # You can modify this based on your criteria for crowd instances
                    
                    # Add segment info if it is a new instance
                    if not any(seg_info["id"] == instance_color_mapping[class_id][color_str] for seg_info in segments_info):
                        segments_info.append({
                        "id": instance_color_mapping[class_id][color_str],
                        "category_id": class_id,
                        "iscrowd": is_crowd
                        })

        record["panoptic_seg"] = {
            "file_name": mask_filename,
            "segments_info": segments_info,
        }

        dataset_dicts.append(record)
        idx += 1
    return dataset_dicts


# --- Register Dataset ---

def register_tree_dataset(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, color_palette):
    """
    Registers the tree dataset with Detectron2.

    Args:
        train_img_dir: Directory containing training RGB images.
        train_mask_dir: Directory containing training RGB panoptic masks.
        val_img_dir: Directory containing validation RGB images.
        val_mask_dir: Directory containing validation RGB panoptic masks.
        color_palette: A list of dictionaries defining the color mapping.
    """
    for d, img_dir, mask_dir in [("train", train_img_dir, train_mask_dir), ("val", val_img_dir, val_mask_dir)]:
        DatasetCatalog.register("tree_" + d, lambda d=d, img_dir=img_dir, mask_dir=mask_dir: get_tree_dicts(img_dir, mask_dir, color_palette))
        MetadataCatalog.get("tree_" + d).set(thing_classes=["non-tree", "tree"])  # Assuming 0: non-tree, 1: tree
        MetadataCatalog.get("tree_" + d).set(panoptic_format="rgb")

    tree_metadata = MetadataCatalog.get("tree_train")
    print(tree_metadata)
    return tree_metadata


def read_color_palette(color_palette_path, invert_to_bgr=False):
        raw_color_palette = pd.read_excel(color_palette_path)  # 4 cols: Index, R, G, B
        rgb_color_palette = raw_color_palette.to_dict(orient='records')
        color_palette = []
        if not invert_to_bgr:
                color_palette = [{list(rgb_color_palette[idx].values())[0]: list(rgb_color_palette[idx].values())[1:]} for idx,itm in enumerate(rgb_color_palette)]
        else:
                for idx,itm in enumerate(rgb_color_palette):
                        values = list(rgb_color_palette[idx].values())          # because values is a dict_values type
                        color_palette.append({values[0]: (values[1:][2], values[1:][1], values[1:][0])})
        return color_palette


# --- Example Usage ---

# Define your color palette - Ensure you have up to 256 unique colors
'''
color_palette = [
    {0: [0, 0, 0]},    # non-tree (black)
    {1: [0, 255, 0]},  # tree (green)
    # ... Add more colors as needed
]
'''
color_palette_path = '/mnt/raid1/dataset/spread/color-palette.xlsx'
color_palette = read_color_palette(color_palette_path, invert_to_bgr=True)


# Define your data directories
dataset_path   = "/mnt/raid1/dataset/spread/spread-femto/"
train_img_dir  = dataset_path + "downtown-europe/rgb"
train_mask_dir = dataset_path + "downtown-europe/instance_segmentation"
val_img_dir    = dataset_path + "rainforest/rgb"
val_mask_dir   = dataset_path + "rainforest/instance_segmentation"

# Register the dataset
tree_metadata = register_tree_dataset(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, color_palette)

# --- Visualize Dataset (Optional) ---
dataset_dicts = DatasetCatalog.get("tree_train")
for d in random.sample(dataset_dicts, 3):  # Visualize 3 random samples
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=tree_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("Visualization", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
sys.exit()

# --- Training Configuration ---

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))  # Example: Panoptic FPN with ResNet-50
cfg.DATASETS.TRAIN = ("tree_train",)
cfg.DATASETS.TEST = ("tree_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")  # Initialize with pre-trained weights
cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.SOLVER.STEPS = []  # Learning rate decay steps (empty = no decay)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.PANOPTIC_FPN.COMBINE.NUM_CLASSES = NUM_CLASSES
cfg.OUTPUT_DIR = "./output"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# --- Training ---

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# --- Inference & Evaluation ---

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # Set threshold for instance segmentation
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.7 # Set threshold for panoptic segmentation
predictor = DefaultPredictor(cfg)

# Example inference on a validation image
dataset_dicts = DatasetCatalog.get("tree_val")
for d in random.sample(dataset_dicts, 1):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=tree_metadata,
                   scale=0.5
    )
    out = v.draw_panoptic_seg_predictions(outputs["panoptic_seg"].to("cpu"), outputs["segments_info"])
    cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Training and inference complete! - written by Gemini-exp-1206")





































fn  = str(root / img_path[0])
img = cv2.imread(fn)
'''
cfg = get_keypoints_cfg()
print(f'Predicting keypoints on {fn}')
outputs = detectron_predict(img, cfg, task='keypoints', window_name='keypoints')
cfg = get_instance_segmentation_cfg()
print(f'Predicting instance segmentation {fn}')
outputs = detectron_predict(img, cfg, task='instance segmentation', window_name='instance segmentation')
'''
cfg = get_panoptic_segmentation_cfg()
print(f'Predicting panoptic segmentation {fn}')
outputs = detectron_predict(img, cfg, task='panoptic segmentation', window_name='panoptic segmentation')
print(f'{outputs = }')
print(f'{outputs["sem_seg"].shape = }')
print(f'{outputs["panoptic_seg"][0].shape = }')
instances = outputs["instances"]
print(f'{instances["pred_boxes"] = }')
print(f'{instances["pred_classes"] = }')
print(f'{instances["pred_masks"] = }')
sys.exit(0)

cfg = get_cfg()
cfg.set_new_allowed(True)
add_maskdino_config(cfg)
cfg.merge_from_file(os.path.join(root,
                                 'MaskDINO/configs/camvid/semantic-segmentation',
                                 'maskdino_R50_bs16_160k_steplr.yaml'))
cfg.MODEL.WEIGHTS = os.path.join(root,
                                 pretrained_models,
                                 'maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth')


dataset_dir = os.getenv("DETECTRON2_DATASETS", "/mnt/raid1/dataset/") # defining the root for the dataset

from detectron2.engine import DefaultTrainer
trainer = DefaultTrainer(cfg)
batch_tr = next(trainer.data_loader.__iter__())
for e in batch_tr:
    print(f'{type(e) = }')
    if isinstance(e, dict):
        for i, key in enumerate(e.keys()):
            if i == 0:
                print(f'\t{key} = {e[key]}')
            else:
                print(f'\t{key}')
#trainer.train()

#%%
catalog = DatasetCatalog
camvid_dataset = None
for dataset in catalog.list():
    if 'camvid' and 'train' in dataset:
        camvid_dataset = dataset
#%%
metadata = MetadataCatalog.get(camvid_dataset)
# %%
'''
Prova a fare un train mediante il simple trainer, costruendo tutte le varie componenti una per una
'''
from maskdino.data.dataset_mappers.mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
from detectron2.data import build_detection_train_loader
from detectron2.engine.train_loop import SimpleTrainer
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
import logging
from detectron2.utils.events import EventStorage, get_event_storage
from tqdm import tqdm
import wandb
import weakref

from detectron2.checkpoint import DetectionCheckpointer

from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess

from format import Text

class MyTrainer(SimpleTrainer):
    def __init__(self, cfg, log=True, zero_grad_before_forward=False):
        
        model = build_model(cfg)
        mapper = MaskFormerSemanticDatasetMapper(cfg)
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
        optimizer = build_optimizer(cfg, model)

        super().__init__(model, data_loader, optimizer, zero_grad_before_forward=False)
        
        self.log = log
        self.classes = self.get_classes_dict(cfg)

        self.checkpointer = DetectionCheckpointer(
            self.model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self)
        )
        
        if cfg.MODEL.WEIGHTS is None:
            self.checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    
    def get_classes_dict(self, cfg):
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        classes = metadata.get('stuff_classes')

        return {k: cls for k, cls in enumerate(classes)}

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)

        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict, outputs = self.model(data)

        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
    
        if self.log:
            wandb.log({'loss': losses, 'iter': self.iter})
            if self.iter % 100 == 0: #Introduce variable in cfg for images to save 
                self.save_images(data, outputs)
        
        losses.backward()

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()
    
    def train(self, start_iter: int = 0, max_iter: int = cfg.SOLVER.MAX_ITER):
        super().train(start_iter, max_iter)

    def before_train(self):
        super().before_train()
        if self.log:
            wandb.init(project='maskdino', config=cfg)

    def after_train(self):
        super().after_train()
        self.checkpointer.save(name=cfg.SAVED_MODEL_NAME)
        if self.log:
            wandb.finish()

    def save_images(self, batched_input: list[dict], outputs: dict) -> None:
        i = random.randint(0, len(batched_input)-1)


        image_dict = batched_input[i]
        image, mask_gt = image_dict['image'], image_dict['sem_seg']

        mask, logits = outputs['pred_masks'][i], outputs['pred_logits'][i]
        mask_cls = F.softmax(logits, dim=-1)[..., :-1]
        mask = mask.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask)
        semseg = F.interpolate(semseg.view(1,semseg.shape[0],semseg.shape[1],semseg.shape[2]),
                            size=(image.shape[1], image.shape[2])).detach()
        
        mask_pred = np.argmax(semseg.cpu(), axis=1)

        masked_img = wandb.Image(image, masks={'prediction': {'mask_data': mask_pred[0].numpy(), 'class_labels': self.classes},
                                            'ground_truth': {'mask_data': mask_gt.numpy(), 'class_labels': self.classes}})
        wandb.log({'masked_image': masked_img})

#%%
'''
Inspect the functions/variables (model -> MaskDINO class):
    model.sem_seg_postprocess_before_inference
    sem_seg_postprocess ('ndo sta???)
    model.semantic_inference
    model.semantic_on (deal with instance_on and panoptic_on)

'''
'''
model = build_model(cfg)

mapper = MaskFormerSemanticDatasetMapper(cfg)
data_loader = build_detection_train_loader(cfg, mapper=mapper)

optimizer = build_optimizer(cfg, model)
'''

# the output directory can be set in 
print(cfg.OUTPUT_DIR) # defaults is ./output

#addind a cfg key for the name of the model to save
cfg.SAVED_MODEL_NAME = "final_model" #find better name

new_trainer = MyTrainer(cfg, log=True, zero_grad_before_forward=True)
new_trainer.train()

from detectron2.evaluation.sem_seg_evaluation import SemSegEvaluator

cfg.MODEL.WEIGHTS = os.path.join(root,
                                 output_dir,
                                 'maskdino_fine_tuned.pth')

model = build_model(cfg)

evaluator = SemSegEvaluator(cfg.DATASETS.TEST[0], cfg.OUTPUT_DIR)
#the path of the images are in:
print(f'{Text(evaluator.input_file_to_gt_file, 'evaluator.input_file_to_gt_file'):inspect}')
# %%
