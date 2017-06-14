# Config file for network training

IMAGE_PATH = "/net/cvcfs/storage/skull-atlas/imgscrape/linear_positive_copy"
SOLVER_PROTO = "../solver.prototxt"
MODEL_DIR = "../models"
PRETRAIN_WEIGHTS_FP = MODEL_DIR + "/bvlc_reference_caffenet.caffemodel"
#PRETRAIN_WEIGHTS_FP = None

GPU = True
SNAPSHOT_ITERS = 10000
MAX_ITERS = 200000
BATCH_SIZE = 30
IM_WIDTH = 227
IM_HEIGHT =  227
