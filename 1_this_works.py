'''
To train there are different possibilities
SimpleTrainer --> don't know how to initialize it from a CfgNode (detectron2/engine/train_loop.py)
DefaultTrainer --> (detectron2/engine/defaults.py)

Problem with cfg.SOLVER.CLIP_GRADIENTS.ENABLED disabling it from the camvid config

the DETECTRON_DATASETS root path (to get the data) must be specified in the script that registers the dataset

'''

'''
The MaskDINO class wants batch_images (a list) containing a dict for each image with keys 
[image, istances, height, width]
'''
#%%
#suppressing the FutureWarning
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from detectron2.config.config import CfgNode
from maskdino.config import add_maskdino_config
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from detectron2.config import get_cfg

root = '/home/giorgio/venvs/MaskDINO'
root_mnt = '/mnt/shared/MaskDINO_stuff'
cfg = get_cfg()
cfg.set_new_allowed(True)
add_maskdino_config(cfg)
cfg.merge_from_file(os.path.join(root,
                                 'MaskDINO/configs/camvid/semantic-segmentation',
                                 'maskdino_R50_bs16_160k_steplr.yaml'))
cfg.MODEL.WEIGHTS = os.path.join(root_mnt,
                                 'MaskDINO_models/downloads',
                                 'maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth')
# Implementing a training session using the simple trainer

# %%
import numpy as np
import wandb, random, time, logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.engine.train_loop import SimpleTrainer
from maskdino.data.dataset_mappers.mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data import MetadataCatalog

class MyTrainer(SimpleTrainer):
    def __init__(self, cfg: CfgNode, log: bool = True): #log variable needs to be eliminated and entered through inside the cfg
        
        model = build_model(cfg)
        mapper = MaskFormerSemanticDatasetMapper(cfg)
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
        optimizer = build_optimizer(cfg, model)

        super().__init__(model, data_loader, optimizer, zero_grad_before_forward=False)
        
        self.log = log
        self.classes = self.get_classes_dict(cfg)
    
    def get_classes_dict(self, cfg: CfgNode):
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
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        
        if self.log:
            wandb.init(project='maskdino', config=cfg)
        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in tqdm(range(start_iter, max_iter)):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
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

trainer = MyTrainer(cfg, log=True)
trainer.train()