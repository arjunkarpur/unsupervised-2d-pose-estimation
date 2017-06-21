
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
    batch_size, feat_len = \
      top_diff.shape[0], top_diff.shape[1]

    diff = np.zeros(bottom[0].diff.shape)
    for i in range(batch_size):
      curr_top_diff = top_diff[i][...].reshape(1,feat_len)
      in_vector = bottom[0].data[i][...]
      jacobian = self.calc_l2norm_jacobian(in_vector)
      diff[i][...] = curr_top_diff.dot(jacobian) # (1x1000)*(1000x1000)
    bottom[0].diff[...] = diff

  def calc_l2norm_jacobian(self, X):
    n = len(X)
    jacobian = np.zeros((n,n))
    for row in range(len(jacobian)):
      for col in range(len(jacobian[0])):
        jacobian[row, col] = self.dFi_Xj(X, row, col)
    return jacobian
  
  def dFi_Xj(self, X, i, j):
    # F_i(X) = X_i/(||X||) = X_i/G(X)
    val = 0
    if i == j:
      val = \
        (self.G_X(X) - (X[i]*self.dG_Xj(X,j))) / \
        (self.G_X(X) * self.G_X(X))
    else:
      val = \
        X[i] * \
        (-1/(self.G_X(X)*self.G_X(X))) * \
        self.dG_Xj(X, j)
    return val

  def G_X(self, X):
    # Vector norm function (G(X) = ||X||)
    return np.sqrt(X.dot(X))
  
  def dG_Xj(self, X, j):
    gradient = \
      0.5 * (1/np.sqrt(X.dot(X))) * (2*X[j])
    return gradient

  def reshape(self, bottom, top):
    pass
