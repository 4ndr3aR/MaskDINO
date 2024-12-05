'''
Loading a model and testing it on a single image using the DefaultPredictor
'''
#%%
import torch
from maskdino.config import add_maskdino_config
import numpy as np 
import os, cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

img_path = ['/mnt/shared/camvid_old/images/0016E5_08141.png',
            '/tmp/macchina.jpg',
            '/tmp/stanza.jpg']

# Following code takes a configuration and creates and create a CfgNode object
root = '/home/giorgio/venvs/MaskDINO'
root_mnt = '/mnt/shared/MaskDINO_stuff'
cfg = get_cfg()
cfg.set_new_allowed(True)
add_maskdino_config(cfg)
cfg.merge_from_file(os.path.join(root,
                                 'configs/ade20k/semantic-segmentation',
                                 'maskdino_R50_bs16_160k_steplr.yaml'))
cfg.MODEL.WEIGHTS = os.path.join(root_mnt,
                                 'MaskDINO_models/downloads',
                                 'maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth')

#accessing to the dataset metadata
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
classes = metadata.stuff_classes
id2name = {id: cls for id, cls in enumerate(classes)} #id to name association

#creating the model and setting it to evaluate
from detectron2.modeling import build_model
model2 = build_model(cfg)
model2.eval()

#the model does not need to be in eval mode!!!
model2.train()
import matplotlib.pyplot as plt
im_2 = cv2.imread(img_path[2])
plt.imshow(im_2)
plt.show()
predictor = DefaultPredictor(cfg)
outputs = predictor(im_2)

masks = outputs['sem_seg'] #masks from the dict output, dim #CLS, H, W
total_pix = 0
for i in range(150):
    mask = masks[i]
    uint8_mask = mask.to(torch.uint8).cpu().numpy()
    rescaled_uint8_mask = uint8_mask * 255
    pixels = len(rescaled_uint8_mask[rescaled_uint8_mask != 0])
    total_pix += pixels
    if pixels != 0:
        print(f'Class: {id2name[i]}')
        plt.imshow(rescaled_uint8_mask)
        plt.show()

print(f'Image covered: {round(total_pix/(masks.shape[1]*masks.shape[2])*100, 2)}')
# %%
'''
DatasetCatalog is a class that allow to register dataset to then be used by maskdino
MetadataCatalog contains metadata of the corresponding datasets
(detectron2/data/catalog.py)
'''
#to look the registered dataset 
from detectron2.data import DatasetCatalog

catalog = DatasetCatalog
for c in catalog.list():
    print(c)

'''
In order to register a different dataset use a script similar to those in maskdino/data/datasets
To use the dataset with a model: create the two corresponding .yaml file.
'''
# %%
