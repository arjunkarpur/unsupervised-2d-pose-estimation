
import caffe

class TripletLossLayer(caffe.Layer):

  def setup(self, bottom, top):
    #TODO
    top[0].reshape(1)
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
