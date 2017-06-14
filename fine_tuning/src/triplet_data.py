
import numpy as np
import caffe
import config

class TripletDataLayer(caffe.Layer):

  def setup(self, bottom, top):
    #TODO

    # Get mean image for subtraction
    mu= np.load('../data/ilsvrc_2012_mean.npy')
    self.mean_imgnet = mu.mean(1).mean(1)
    print 'mean-subtracted values:', zip('BGR', self.mean_imgnet)
    pass

  def forward(self, bottom, top):
    #TODO
    pass
  
  def backward(self, top, prop_down, bottom):
    #TODO
    pass

  def reshape(self, bottom, top):
    #TODO
    pass
