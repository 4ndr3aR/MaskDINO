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

import torch
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

import wandb
from tqdm import tqdm
import time
from maskdino.data.dataset_mappers.mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
from detectron2.data import build_detection_train_loader
from detectron2.engine.train_loop import SimpleTrainer
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
import logging
from detectron2.utils.events import EventStorage, get_event_storage

# implementation of the SimpleTrainer
# in the class the run_step and train function are overwritten to add tqdm and wandb logging
class MyTrainer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, zero_grad_before_forward=False):
        super().__init__(model, data_loader, optimizer, zero_grad_before_forward=False)
    
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
        loss_dict = self.model(data)
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

        wandb.log({'loss': losses, 'iter': self.iter})
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

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

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
                wandb.finish()

model = build_model(cfg)

mapper = MaskFormerSemanticDatasetMapper(cfg)
data_loader = build_detection_train_loader(cfg, mapper=mapper)

optimizer = build_optimizer(cfg, model)

new_trainer = MyTrainer(model, data_loader, optimizer, zero_grad_before_forward=True)

new_trainer.train(0, cfg.SOLVER.MAX_ITER)
# %%
