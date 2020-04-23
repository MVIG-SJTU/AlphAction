from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()

_C.MODEL.WEIGHT = ""

# -----------------------------------------------------------------------------
# ROI action head config.
# -----------------------------------------------------------------------------
_C.MODEL.ROI_ACTION_HEAD = CN()

# Feature extractor config.
_C.MODEL.ROI_ACTION_HEAD.FEATURE_EXTRACTOR = "MaxpoolFeatureExtractor"

# Config for pooler in feature extractor.
# Pooler type, can be 'pooling3d' or 'align3d'
_C.MODEL.ROI_ACTION_HEAD.POOLER_TYPE = 'align3d'
_C.MODEL.ROI_ACTION_HEAD.POOLER_RESOLUTION = 7
_C.MODEL.ROI_ACTION_HEAD.POOLER_SCALE = 1./16
# Only used for align3d
_C.MODEL.ROI_ACTION_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER = False

_C.MODEL.ROI_ACTION_HEAD.MLP_HEAD_DIM = 1024

# Action predictor config.
_C.MODEL.ROI_ACTION_HEAD.PREDICTOR = "FCPredictor"
_C.MODEL.ROI_ACTION_HEAD.DROPOUT_RATE = 0.0
_C.MODEL.ROI_ACTION_HEAD.NUM_CLASSES = 80

# Action loss evaluator config.
_C.MODEL.ROI_ACTION_HEAD.PROPOSAL_PER_CLIP = 10
_C.MODEL.ROI_ACTION_HEAD.NUM_PERSON_MOVEMENT_CLASSES = 14
_C.MODEL.ROI_ACTION_HEAD.NUM_OBJECT_MANIPULATION_CLASSES = 49
_C.MODEL.ROI_ACTION_HEAD.NUM_PERSON_INTERACTION_CLASSES = 17

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = 256
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 464
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 256
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 464
# Values to be used for image normalization, in rgb order
_C.INPUT.PIXEL_MEAN = [122.7717, 115.9465, 102.9801]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [57.375, 57.375, 57.375]
# Convert image to BGR format (for Caffe2 models)
_C.INPUT.TO_BGR = False

_C.INPUT.FRAME_NUM = 32
_C.INPUT.FRAME_SAMPLE_RATE = 2
_C.INPUT.TAU = 8
_C.INPUT.ALPHA = 8

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of dataset loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 16
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# Available backbone conv-body should be registered in modeling.backbone.backbone.py
_C.MODEL.BACKBONE.CONV_BODY = "Slowfast-Resnet50"

_C.MODEL.BACKBONE.FROZEN_BN = False

_C.MODEL.BACKBONE.BN_MOMENTUM = 0.1
_C.MODEL.BACKBONE.BN_EPSILON = 1e-05

# Kaiming:
# We may use 0 to initialize the residual branch of a residual block,
# so the inital state of the block is exactly identiy. This helps optimizaiton.
_C.MODEL.BACKBONE.BN_INIT_GAMMA = 0.0

_C.MODEL.BACKBONE.I3D = CN()
_C.MODEL.BACKBONE.I3D.CONV3_NONLOCAL = True
_C.MODEL.BACKBONE.I3D.CONV4_NONLOCAL = True
_C.MODEL.BACKBONE.I3D.CONV3_GROUP_NL = False

# ---------------------------------------------------------------------------- #
# Slowfast options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE.SLOWFAST = CN()
_C.MODEL.BACKBONE.SLOWFAST.BETA = 1./8
_C.MODEL.BACKBONE.SLOWFAST.LATERAL = 'tconv'
_C.MODEL.BACKBONE.SLOWFAST.SLOW = CN()
_C.MODEL.BACKBONE.SLOWFAST.SLOW.ACTIVE = True
_C.MODEL.BACKBONE.SLOWFAST.SLOW.CONV3_NONLOCAL = True
_C.MODEL.BACKBONE.SLOWFAST.SLOW.CONV4_NONLOCAL = True
_C.MODEL.BACKBONE.SLOWFAST.SLOW.CONV3_GROUP_NL = False
_C.MODEL.BACKBONE.SLOWFAST.FAST = CN()
_C.MODEL.BACKBONE.SLOWFAST.FAST.ACTIVE = True
_C.MODEL.BACKBONE.SLOWFAST.FAST.CONV3_NONLOCAL = False
_C.MODEL.BACKBONE.SLOWFAST.FAST.CONV4_NONLOCAL = False
_C.MODEL.BACKBONE.SLOWFAST.FAST.CONV3_GROUP_NL = False

# ---------------------------------------------------------------------------- #
# Nonlocal options
# ---------------------------------------------------------------------------- #
_C.MODEL.NONLOCAL = CN()
_C.MODEL.NONLOCAL.CONV_INIT_STD = 0.01
_C.MODEL.NONLOCAL.USE_ZERO_INIT_CONV = False
_C.MODEL.NONLOCAL.NO_BIAS = False
_C.MODEL.NONLOCAL.USE_MAXPOOL = True
_C.MODEL.NONLOCAL.USE_SOFTMAX = True
_C.MODEL.NONLOCAL.USE_SCALE = True

_C.MODEL.NONLOCAL.USE_BN = True
_C.MODEL.NONLOCAL.FROZEN_BN = False

_C.MODEL.NONLOCAL.BN_MOMENTUM = 0.1
_C.MODEL.NONLOCAL.BN_EPSILON = 1e-05
_C.MODEL.NONLOCAL.BN_INIT_GAMMA = 0.0


_C.IA_STRUCTURE = CN()
_C.IA_STRUCTURE.ACTIVE = False
_C.IA_STRUCTURE.STRUCTURE = "serial"
_C.IA_STRUCTURE.MAX_PER_SEC = 5
_C.IA_STRUCTURE.MAX_PERSON = 25
_C.IA_STRUCTURE.DIM_IN = 2304
_C.IA_STRUCTURE.DIM_INNER = 512
_C.IA_STRUCTURE.DIM_OUT = 512
_C.IA_STRUCTURE.LENGTH = (30, 30)
_C.IA_STRUCTURE.MEMORY_RATE = 1
_C.IA_STRUCTURE.FUSION = "concat"
_C.IA_STRUCTURE.CONV_INIT_STD = 0.01
_C.IA_STRUCTURE.DROPOUT = 0.
_C.IA_STRUCTURE.NO_BIAS = False
_C.IA_STRUCTURE.I_BLOCK_LIST = ['P', 'O', 'M', 'P', 'O', 'M']
_C.IA_STRUCTURE.LAYER_NORM = False
_C.IA_STRUCTURE.TEMPORAL_POSITION = True
_C.IA_STRUCTURE.ROI_DIM_REDUCE = True
_C.IA_STRUCTURE.USE_ZERO_INIT_CONV = True
_C.IA_STRUCTURE.MAX_OBJECT = 0

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of video clips per batch
# This is global, so if we have 8 GPUs and VIDEOS_PER_BATCH = 16, each GPU will
# see 2 clips per batch
_C.TEST.VIDEOS_PER_BATCH = 16

# Config used in inference.
_C.TEST.EXTEND_SCALE = (0.1, 0.05)
_C.TEST.BOX_THRESH = 0.8
_C.TEST.ACTION_THRESH = 0.05

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
