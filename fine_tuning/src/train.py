
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import config

class SolverWrapper:
  
  def __init__(self, solver_proto, model_dir, pretrain_weights=None):
    # Set to use CPU/GPU
    if config.GPU:
      caffe.set_mode_gpu()
      #caffe.set_device(0)
    else:
      caffe.set_mode_cpu()

    # Initialize solver and net
    self.model_dir = model_dir
    self.solver = caffe.SGDSolver(solver_proto)
    if pretrain_weights is not None:
      self.solver.net.copy_from(pretrain_weights)

    # Set solver params
    self.solver_param = caffe_pb2.SolverParameter()
    with open(solver_proto, 'rt') as f:
      pb2.text_format.Merge(f.read(), self.solver_param)

  def train(self, max_iters):
    #TODO: print more info?
    while self.solver.iter < max_iters:
      self.solver.step(1)
    pass
  
# Begin training
print "Initializing solver..."
solv = SolverWrapper(config.SOLVER_PROTO, \
                     config.MODEL_DIR,    \
                     config.PRETRAIN_WEIGHTS_FP)
print "Beginning training..."
solv.train(config.MAX_ITERS)
print "Training finished!"
