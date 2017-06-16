
import caffe
import config
import numpy as np

class TripletLossLayer(caffe.Layer):

  def setup(self, bottom, top):
    self.batch_size = config.BATCH_SIZE
    assert (self.batch_size%3 == 0)
    self.triplets_per_batch = self.batch_size / 3
    top[0].reshape(1)
    self.margin = 1
    self.neg_loss_triplets = set()

  def forward(self, bottom, top):
    loss = 0
    self.neg_loss_triplets = set()
    for i in range(self.triplets_per_batch):
      curr_anchor = bottom[0].data[i*3]
      curr_pos = bottom[0].data[(i*3)+1]
      curr_neg = bottom[0].data[(i*3)+2]

      curr_loss = \
        np.dot((curr_anchor - curr_pos),(curr_anchor - curr_pos)) - \
        np.dot((curr_anchor - curr_neg),(curr_anchor - curr_neg)) + \
        self.margin
      if curr_loss < 0:
        curr_loss = 0
        self.neg_loss_triplets.add(i)
      loss += curr_loss
    top[0].data[...] = loss
  
  def backward(self, top, propagate_down, bottom):

    diff = np.zeros_like(bottom[0].data)

    for i in range(self.triplets_per_batch):
      if i in self.neg_loss_triplets:
        continue
      a_i = bottom[0].data[i*3]
      p_i = bottom[0].data[(i*3)+1]
      n_i = bottom[0].data[(i*3)+2]
      d_a_i = 2*(n_i - p_i)
      d_p_i = 2*(p_i - a_i)
      d_n_i = 2*(a_i - n_i)
      diff[i*3] = d_a_i
      diff[(i*3)+1] = d_p_i
      diff[(i*3)+2] = d_n_i
    bottom[0].diff[...] = diff * propagate_down

  def reshape(self, bottom, top):
    #TODO
    pass
