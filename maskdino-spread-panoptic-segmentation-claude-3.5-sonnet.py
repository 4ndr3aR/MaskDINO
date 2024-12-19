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












from PIL import Image

# Written by Claude-3.5-sonnet

class TreeSegmentationDataset:
    def __init__(self, img_dir, mask_dir, color_palette):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # Convert color palette from list of dicts to a more efficient format
        self.color_palette = {}
        for d in color_palette:
            for k, v in d.items():
                self.color_palette[tuple(v)] = k

    def rgb_to_instance_id(self, mask):
        """Convert RGB mask to instance ID mask"""
        # Convert mask to (H, W, 3) if it's not already
        if len(mask.shape) == 2 or mask.shape[2] != 3:
            raise ValueError("Mask must be RGB")
        
        h, w = mask.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.int32)
        
        # Create a view of the mask as a structured array for efficient color matching
        mask_view = mask.view(dtype=np.dtype([('', np.uint8)] * 3)).reshape(h, w)
        
        # Map RGB colors to instance IDs
        for rgb, instance_id in self.color_palette.items():
            match_mask = mask_view == np.array(rgb, dtype=np.uint8).view(mask_view.dtype)
            instance_mask[match_mask] = instance_id
            
        return instance_mask

    def get_class_id(self, instance_id):
        """Return class ID (0 for non-tree, 1 for tree) based on instance ID"""
        # This is an example - modify according to your actual class mapping
        return 1 if instance_id > 0 else 0

    def get_dataset_dicts(self):
        dataset_dicts = []
        
        # Get list of image files
        img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        for idx, img_file in enumerate(img_files):
            record = {}
            
            # Image filename
            image_path = os.path.join(self.img_dir, img_file)
            mask_path = os.path.join(self.mask_dir, img_file)  # Assuming same filename for mask
            
            # Read image for height and width
            img = Image.open(image_path)
            width, height = img.size
            
            record["file_name"] = image_path
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            
            # Read and process mask
            mask = np.array(Image.open(mask_path))
            instance_mask = self.rgb_to_instance_id(mask)
            
            # Find unique instances (excluding background)
            instance_ids = np.unique(instance_mask)
            instance_ids = instance_ids[instance_ids != 0]  # Remove background
            
            annotations = []
            
            for instance_id in instance_ids:
                # Create binary mask for this instance
                binary_mask = (instance_mask == instance_id)
                
                # Find bounding box
                y_indices, x_indices = np.nonzero(binary_mask)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                    
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # Convert binary mask to RLE
                binary_mask = np.asfortranarray(binary_mask)
                from pycocotools import mask as mask_util
                rle = mask_util.encode(binary_mask)
                
                annotation = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": rle,
                    "category_id": self.get_class_id(instance_id),
                    "instance_id": int(instance_id)
                }
                annotations.append(annotation)
            
            record["annotations"] = annotations
            dataset_dicts.append(record)
            
        return dataset_dicts

def register_tree_dataset(name, img_dir, mask_dir, color_palette):
    """Register the dataset with Detectron2"""
    dataset = TreeSegmentationDataset(img_dir, mask_dir, color_palette)
    DatasetCatalog.register(name, dataset.get_dataset_dicts)
    MetadataCatalog.get(name).set(
        thing_classes=["non-tree", "tree"],
        thing_colors=[[0, 0, 0], [0, 255, 0]]  # You can modify these colors
    )




























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
