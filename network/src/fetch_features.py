
import os
import sys
import h5py
import caffe
import numpy as np

# Input vars
data_out_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/feature_data/finetune/rendered_images.hdf5"
file_list_fp = "../../inputs/rend_dirs.txt"
dir_base = "/mnt/localscratch/arjun/renderings_210/healthy"
feat_type = 'RENDERED' # or REAL
fetch_layer = 'embedding'
num_batches = 50
num_ims_per_model = 210

#####################
#  BEGIN FUNCTIONS  #
#####################

def initialize_net():
  # Create network
  caffe.set_mode_gpu()
  model_def = "../deploy.prototxt"
  model_weights = "../models/finetune_posenet_iter_10299.caffemodel"
  net = caffe.Net(model_def, model_weights, caffe.TEST)
  return net

def create_input_transformer(net):
  # Load mean ImageNet image
  mu = np.load('../data/ilsvrc_2012_mean.npy')
  mu = mu.mean(1).mean(1)

  # Create transformer for input image
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2,0,1))
  transformer.set_mean('data', mu)
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2,1,0))
  return transformer

def fetch_features_real(net, transformer, filenames, grp):
  total_num_images = len(filenames)
  batch_size = total_num_images / num_batches
  print 'Fetching features for %i real images' % total_num_images

  # Split into batches
  for i in range(num_batches):
    print "Running batch %i / %i \t %d %%" % \
      (i+1, num_batches, float(100) * float((float(i+1)/float(num_batches))))
    start = i * batch_size
    if i == num_batches - 1:
      end = total_num_images
    else:
      end = start + batch_size
    num_images = end-start

    # Load input images
    net.blobs['data'].reshape(num_images, 3, 227, 227)
    for i in range(start, end):
      image = caffe.io.load_image(filenames[i])
      transformed_image = transformer.preprocess('data', image)
      net.blobs['data'].data[i-start] = transformed_image

    # Run feed-forward and get features
    output = net.forward(blobs=[fetch_layer])
    features = {}
    for i in range(start, end):
      curr_features = output[fetch_layer][i-start]
      features[filenames[i]] = curr_features

    # Write to output file
    for fp, feature_v in features.items():
      name = fp.split("/")[-1]
      dset = grp.create_dataset(name, (len(feature_v),), dtype='d')
      dset[...] = feature_v
  return

def fetch_features_rendered(net, transformer, dirnames, base_grp):

  # Run per model
  total_num_models = len(dirnames)
  print 'Fetching features for %i models' % total_num_models
  for i in range(total_num_models):
    ims = []
    curr_dir = dirnames[i]
    print "%s \t %i/%i" % (curr_dir, i+1, total_num_models)
    for z in range(num_ims_per_model):
      ims.append(os.path.join(dir_base, curr_dir, "%i.png" % z))

    # Load all 210 images for a given model
    net.blobs['data'].reshape(len(ims),3,227,227)
    for z in range(len(ims)):
      image = caffe.io.load_image(ims[z])
      transformed_image = transformer.preprocess('data', image)
      net.blobs['data'].data[z] = transformed_image

    # Feed forward and get feature data
    output = net.forward(blobs=[fetch_layer])
    features = []
    for z in range(len(ims)):
      curr_features = output[fetch_layer][z]
      features.append(curr_features)

    # Create HDF5 group for each model and write feature data
    grp = base_grp.create_group(curr_dir)
    for z in range(len(ims)):
      name = "%i.png" % z
      feature_v = features[z]
      dset = grp.create_dataset(name, (len(feature_v),), dtype='d')
      dset[...] = feature_v
  return

#####################
#   END FUNCTIONS   #
#####################

if __name__ == "__main__":

  # Load input file
  filenames = np.loadtxt(file_list_fp, str, delimiter='\n')

  # Create network and init output file
  net = initialize_net()
  transformer = create_input_transformer(net)
  f = h5py.File(data_out_fp, 'w')
  grp = f.create_group('imagefeatures')
  
  # Fetch features
  if feat_type == 'REAL':
    fetch_features_real(net, transformer, filenames, grp)
  elif feat_type == 'RENDERED':
    fetch_features_rendered(net, transformer, filenames, grp)

  # Close output file
  print "Finished!"
  f.close()
