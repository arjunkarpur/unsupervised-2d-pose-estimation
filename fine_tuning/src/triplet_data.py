
import numpy as np
import caffe
import config
import os

class TripletDataLayer(caffe.Layer):

  def setup(self, bottom, top):

    # Load mean image for subtraction
    mu= np.load('../data/ilsvrc_2012_mean.npy')
    self.mean_imgnet = mu.mean(1).mean(1)
    print 'Mean-subtracted values:', zip('BGR', self.mean_imgnet)

    # Set shape of layer
    self.batch_size = config.BATCH_SIZE
    assert(self.batch_size%3 == 0)
    self.triplets_per_batch = self.batch_size / 3
    top[0].reshape(self.batch_size, 3, config.IM_WIDTH, config.IM_HEIGHT)
    #top[1].reshape(self.batch_size)

    # Image file path bases
    self.real_im_path = config.REAL_IMAGE_PATH
    self.rend_im_path = config.RENDERED_IMAGE_PATH

    # Create transformer for input images
    self.transformer = caffe.io.Transformer({'data': top[0].data.shape})
    self.transformer.set_transpose('data', (2,0,1))
    self.transformer.set_mean('data', self.mean_imgnet)
    self.transformer.set_raw_scale('data', 255)
    self.transformer.set_channel_swap('data', (2,1,0))

    # Load triplets from file (TODO: temp)
    triplets_fp = "../../triplet_generation/out/triplets_shuffle.txt"
    triplets_f = open(triplets_fp, 'r')
    self.triplets = triplets_f.readlines()
    triplets_f.close()
    self.batch_num = 0

  def forward(self, bottom, top):
    top[0].data[...] = self.get_next_batch()
  
  def backward(self, top, prop_down, bottom):
    # No backprop for data layer
    pass

  def reshape(self, bottom, top):
    #TODO
    pass

  def get_next_batch(self):

    #TODO: do more inteligently: mini batch hard positive/negatives

    # Get the next BATCH_SIZE triplets
    base = self.triplets_per_batch*self.batch_num
    end = self.triplets_per_batch*(self.batch_num + 1)
    triplets_sample = self.triplets[base:end]
    self.batch_num += 1

    # Assemble list of image filepaths to load and preprocess
    im_fp_list = []
    for t in triplets_sample:
      split = (t.split("\n")[0]).split(",")
      anchor_name = split[0]
      pos_name = split[1]
      neg_name = split[2]

      anchor_fp = os.path.join(self.real_im_path, anchor_name)
      pos_name_split = pos_name.split("_")
      pos_fp = os.path.join(self.rend_im_path, "_".join(pos_name_split[1:]), (pos_name_split[0])[4:]+".png")
      neg_name_split = neg_name.split("_")
      neg_fp = os.path.join(self.rend_im_path, "_".join(neg_name_split[1:]), (neg_name_split[0])[4:]+".png")
      im_fp_list.append(anchor_fp)
      im_fp_list.append(pos_fp)
      im_fp_list.append(neg_fp)

    # Load and preprocess
    images_blob = []
    for im_fp in im_fp_list:
      im = caffe.io.load_image(im_fp)
      transformed_im = self.transformer.preprocess('data', im)
      images_blob.append(transformed_im)
    return images_blob
