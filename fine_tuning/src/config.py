# Config file for network training

# Filepaths
REAL_IMAGE_PATH = "/mnt/localscratch/arjun/real_image_dump"
RENDERED_IMAGE_PATH = "/mnt/localscratch/arjun/renderings_210/healthy"
SOLVER_PROTO = "../solver.prototxt"
PRETRAIN_WEIGHTS_FP = "../models/bvlc_reference_caffenet.caffemodel"

# Network parameters
GPU = True
BATCH_SIZE = 30
IM_WIDTH = 227
IM_HEIGHT =  227
