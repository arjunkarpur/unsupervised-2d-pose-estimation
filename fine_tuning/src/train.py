
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import config

def create_sgd_solver(solver_proto, pretrain_weights=None):

  # Set to use CPU/GPU
  if config.GPU:
    caffe.set_mode_gpu()
    #caffe.set_device(0)
  else:
    caffe.set_mode_cpu()

  # Initialize solver and net
  solver = caffe.SGDSolver(solver_proto)
  if pretrain_weights is not None:
    solver.net.copy_from(pretrain_weights)

  # Set solver params
  solver_param = caffe_pb2.SolverParameter()
  with open(solver_proto, 'rt') as f:
    pb2.text_format.Merge(f.read(), solver_param)

  # Return SGD Solver object
  return solver
 
# Begin training
print "Initializing solver..."
sgd_solver = create_sgd_solver(config.SOLVER_PROTO, \
                         config.PRETRAIN_WEIGHTS_FP)
print "Beginning training..."
sgd_solver.solve()
print "Training finished!"
