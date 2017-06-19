
import caffe
import numpy as np
from sklearn import preprocessing

class L2NormLayer(caffe.Layer):
  
  def setup(self, bottom, top):
    shape = np.shape(bottom[0].data)
    top[0].reshape(shape[0], shape[1])

  def forward(self, bottom, top):
    count = 0
    for embedding in bottom[0].data:
      normalized_embedding = \
        preprocessing.normalize(embedding.reshape(1, -1), norm='l2')
      top[0].data[count][...] = normalized_embedding
      count += 1
      
  def backward(self, top, propagate_down, bottom):
    # No backwards prop
    bottom[0].diff[...] = np.ones(np.shape(bottom[0].diff))

  def reshape(self, bottom, top):
    pass
