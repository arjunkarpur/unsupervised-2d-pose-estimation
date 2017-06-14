
import numpy as np
import caffe
import config

class TripletDataLayer(caffe.Layer):

  def setup(self, bottom, top):

    # Load mean image for subtraction
    mu= np.load('../data/ilsvrc_2012_mean.npy')
    self.mean_imgnet = mu.mean(1).mean(1)
    print 'mean-subtracted values:', zip('BGR', self.mean_imgnet)

    # Set shape of layer
    self.batch_size = config.BATCH_SIZE
    top[0].reshape(self.batch_size, 3, config.IM_WIDTH, config.IM_HEIGHT)
    #top[1].reshape(self.batch_size)

    #TODO
    pass

  def forward(self, bottom, top):
    #TODO
    pass
  
  def backward(self, top, prop_down, bottom):
    # No backprop for data layer
    pass

  def reshape(self, bottom, top):
    #TODO
    pass
