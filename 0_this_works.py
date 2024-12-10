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

#creating the dataloader
from maskdino.data.dataset_mappers.mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
from detectron2.data import build_detection_train_loader

mapper_old = MaskFormerSemanticDatasetMapper(cfg) # takes an dict of image input and returns a dict
# to get the dataloader taken this below from train_net.py
data_loader_old = build_detection_train_loader(cfg, mapper=mapper_old) #returns the dataloader
batch = next(data_loader_old.__iter__())
# the batch is a list of dict, one for every images
# following: batch content
# each image is a dict with: file_path, img dimentions, image, mask
for e in batch:
    print(f'{type(e) = }')
    if isinstance(e, dict):
        for i, key in enumerate(e.keys()):
            if i == 0:
                print(f'\t{key} = {e[key]}')
            else:
                print(f'\t{key}')

import matplotlib.pyplot as plt
image = batch[0]
fig, ax = plt.subplots(1,2)
ax[0].imshow(image['image'].permute(1,2,0))
ax[1].imshow(image['sem_seg'])

# don't really know what is inside it, but it appears that the DefaultTrainer want's it 
instances = image['instances'].get_fields()