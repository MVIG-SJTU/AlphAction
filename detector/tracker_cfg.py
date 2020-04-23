from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = '../detector/tracker/cfg/yolov3.cfg'
cfg.WEIGHTS = '../data/models/detector_models/jde.uncertainty.pt'
cfg.IMG_SIZE =  (1088, 608)
cfg.NMS_THRES =  0.4
cfg.CONFIDENCE = 0.2
cfg.BUFFER_SIZE = 30 # frame buffer
