
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
    batch_size = top[0].diff.shape[0]

    diff = np.zeros(bottom[0].diff.shape)
    for i in range(batch_size):
      diff[i] = \
        self.calc_l2_norm_gradient( \
          bottom[0].data[i], top[0].data[i], top[0].diff[i] \
        )
    bottom[0].diff[...] = diff

  def reshape(self, bottom, top):
    shape = np.shape(bottom[0].data)
    top[0].reshape(shape[0], shape[1])

  def calc_l2_norm_gradient(self, v, v_bar, dFdvbar):
    # See 'network/l2_norm_gradient.jpg' for formulation
    v = v.reshape(v.size, 1)
    v_bar = v_bar.reshape(v_bar.size,1)
    v_bar_t = v_bar.reshape(1, v_bar.size)
    dFdvbar = dFdvbar.reshape(dFdvbar.size,1)

    num_one = dFdvbar
    num_two = v_bar.dot(v_bar_t.dot(dFdvbar))
    denom = np.sqrt(v.transpose().dot(v))
    result = (num_one - num_two)/denom
    return result.reshape(1, v_bar.size)
