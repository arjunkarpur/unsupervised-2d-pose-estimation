
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
        preprocessing.normalize((embedding.astype('float64')).reshape(1, -1), norm='l2')
      top[0].data[count][...] = normalized_embedding
      count += 1
      
  def backward(self, top, propagate_down, bottom):
    top_diff = top[0].diff[...]
    diff = np.zeros(bottom[0].diff.shape)
    #TODO: do stuff
    bottom[0].diff[...] = diff

  def reshape(self, bottom, top):
    pass
