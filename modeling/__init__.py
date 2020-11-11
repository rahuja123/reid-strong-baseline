# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .ResNet50_CBAM_TwinTail_v5 import ResNet50_CBAM_TwinTail_v5

#
# def build_model(cfg, num_classes):
#     # if cfg.MODEL.NAME == 'resnet50':
#     #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
#     if cfg.SOLVER.MULTITASK_CAM_LOSS.CAMS > 0:
#         model = getattr(models, cfg.MODEL.NAME)(num_classes,cfg.SOLVER.MULTITASK_CAM_LOSS.CAMS, cfg.MODEL.LAST_STRIDE, cfg.MODEL.POOL, cfg.MODEL.BNNECK, cfg.SOLVER.MULTITASK_CAM_LOSS.REVERSE_GRAD, cfg.MODEL.INSTANCE_NORM)
#     else:
#         model = getattr(models, cfg.MODEL.NAME)(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.POOL, cfg.MODEL.BNNECK)
#     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
#     return model
