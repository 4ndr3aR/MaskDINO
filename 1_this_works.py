'''
To train there are different possibilities
SimpleTrainer --> don't know how to initialize it from a CfgNode (detectron2/engine/train_loop.py)
DefaultTrainer --> (detectron2/engine/defaults.py)

Problem with cfg.SOLVER.CLIP_GRADIENTS.ENABLED
remuving it from the camvid config
'''