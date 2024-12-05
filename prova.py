#!/usr/bin/env python3

#%%
import torch, detectron2, maskdino
from maskdino.config import add_maskdino_config
import numpy as np 
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from fastai.vision.all import *
# %%

root = '/home/giorgio/venvs/MaskDINO'
root_mnt = '/mnt/shared/MaskDINO_stuff'
img_path = ['/mnt/shared/camvid_old/images/0016E5_08141.png', '/tmp/macchina.jpg']
cfg = get_cfg()
cfg.set_new_allowed(True)
add_maskdino_config(cfg)
cfg.merge_from_file(os.path.join(root,
                                 'configs/camvid/semantic-segmentation',
                                 'maskdino_R50_bs16_160k_steplr.yaml'))
cfg.MODEL.WEIGHTS = os.path.join(root_mnt,
                                 'MaskDINO_models/downloads',
                                 'maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth')


#%%
model_specifics = maskdino.maskdino.MaskDINO.from_config(cfg)
metadata = model_specifics['metadata']
classes = metadata.stuff_classes
id2name = {id: cls for id, cls in enumerate(classes)}

#%%
from detectron2.modeling import build_model
model2 = build_model(cfg)

model2.eval()

model2_components = list(model2.children())
comp2 = model2_components[1]
comp2_components = list(comp2.children())
# %%
test = False
if test:
    print(f'{model2 = }')
    print(f'{model2.__dict__ = }')
    print(f'{model2_components = }')
    #print(f'{model2_components.__dict__ = }')
    print(f'{comp2 = }')
    print(f'{comp2.__dict__ = }')
    print(f'{comp2_components = }')
    print(f'{model2.sem_seg_head.predictor.__dict__ = }')
    #print(f'{comp2_components.__dict__ = }')
    for key in model2.keys():
        print(f'{key}: {model2[key]}')

#%%
class DictTensorConverter(nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = None
        pass

    def forward(self, x):
        print(f'{type(x) = }')
        if isinstance(x, dict):
            self.keys = x.keys()
        if 'sem_seg' in self.keys:
            return x['sem_seg']
        else:
            print('Invalid keys')
            for key in self.keys:
                print(key)

# %%
names = ['backbone', 'sem_seg_head', 'criterion']
final_model_v4 = nn.Sequential(OrderedDict((name, component) for name, component in zip(names[:-1], model2_components[:-1])))
# %%
im = torch.from_numpy(cv2.imread(img_path[0]))/255
sz = im.shape
batch = im.permute(2,0,1).view(1,sz[2],sz[0],sz[1]).cuda()
out_new = final_model_v4(batch)
print(len(out_new))
out_dict = out_new[0] # i due elementi della tupla sono un dict e un non type

#%%
for key in out_dict.keys():
    if isinstance(out_dict[key], torch.Tensor):
        print(f'{key}: {out_dict[key].shape}')
    else:
        print(f'{key}: {len(out_dict[key])}')

#%%
#the model does not need to be in eval mode!!!
model2.train()
import matplotlib.pyplot as plt
im_2 = cv2.imread(img_path[1])
plt.imshow(im_2)
plt.show()
predictor = DefaultPredictor(cfg)
outputs = predictor(im_2)

masks = outputs['sem_seg']
total_pix = 0
for i in range(13):
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
#%%

dataset_dir = os.getenv("DETECTRON2_DATASETS", "/mnt/shared") # defining the root for the dataset

from detectron2.engine import DefaultTrainer
trainer = DefaultTrainer(cfg)

# %%
