
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import config

GPU = True
solver_proto = "../solver.prototxt"
model_dir = "../models"
pretrain_weights = model_dir + "/bvlc_reference_alexnet.caffemodel"

class SolverWrapper:
  
  def __init__(self, solver_proto, model_dir, pretrain_weights):
    # Set to use CPU/GPU
    if GPU:
      caffe.set_mode_gpu()
      caffe.set_device(0)
    else:
      caffe.set_mode_cpu()

    # Initialize solver and net
    self.model_dir = model_dir
    self.solver = caffe.SGDSolver(solver_proto)
    self.solver.net.copy_from(pretrain_weights)

    # Set solver params
    self.solver_param = caffe_pb2.SolverParameter()
    with open(solver_proto, 'rt') as f:
      pb2.text_format.Merge(f.read(), self.solver_param)

  def train(max_iters):
    #TODO: print more info?
    while self.solver.iter < max_iters:
      self.solver.step(1)
    pass
  
# Begin training
print "Initializing solver..."
solv = SolverWrapper(solver_proto, model_dir, pretrain_weights)
print "Beginning training..."
solv.train(config.MAX_ITERS)
print "Training finished!"
