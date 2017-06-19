import numpy
import h5py

# Input vars
images_datapath = "/net/cvcfs/storage/skull-atlas/imgscrape/feature_data/finetune/real_images_fc8.hdf5"
renderings_datapath = "/net/cvcfs/storage/skull-atlas/imgscrape/feature_data/finetune/renderings_images_fc8.hdf5"
imlist_fp = "../inputs/split/7.txt"
data_out_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/dist_grids/finetune/finetune_dist_grid_7.hdf5"

# Open data files for reading
renderings_f = h5py.File(renderings_datapath, 'r')
images_f = h5py.File(images_datapath, 'r')
renderings_grp = renderings_f["imagefeatures"]
images_grp = images_f["imagefeatures"]

# Determine number of models and renderings per model
num_models = len(renderings_grp)
num_poses = 0
for model_name in renderings_grp:
  num_poses = len(renderings_grp[model_name])
  break
print "Num models: %i" % num_models
print "Num views per model: %i" % num_poses

# Read list of files to find L2 distances
im_list = []
imlist_f = open(imlist_fp, 'r')
lines = imlist_f.readlines()
imlist_f.close()
for l in lines:
  im_list.append(l.split("\n")[0])

# Open output hdf5 file
data_out_f = h5py.File(data_out_fp, 'w')

#####################
#  BEGIN FUNCTIONS  #
#####################

def tuple_sort(one, two):
  return int(one[1] - two[1])

def L2(descrip_one, descrip_two):
  if len(descrip_one) != len(descrip_two):
    print "L2 ERROR!"
    return -1
  dist = 0
  for i in range(len(descrip_one)):
    dist += ((descrip_one[i] - descrip_two[i]) * \
             (descrip_one[i] - descrip_two[i]))
  return dist

def calc_distance_grid(descriptor, renderings_grp):
  # 210 (poses) x 90 (models)
  dist_grid = \
    [[('',0) for x in range(num_models)] for y in range(num_poses)]

  # For each pose, calc distance between image and each model in that pose
  for i in range(num_poses):
    pose_row = []
    for model_name in renderings_grp:
      im_name = "%i.png" % i
      curr_d = renderings_grp[model_name][im_name][:]
      pose_row.append([str(model_name), float(L2(descriptor,curr_d))])
    sorted_pose_row = sorted(pose_row, key=lambda x:x[1])
    dist_grid[i] = sorted_pose_row
  return dist_grid

#####################
#   END FUNCTIONS   #
#####################

print "Calculating distance grid for %i images" % len(im_list)
count = 0
for im in im_list:
  count += 1
  print "Calculating grid: %i / %i" % (count, len(im_list))
  if im not in images_grp:
    print "ERROR: No descriptor data available for image %s" % im
    continue
  descriptor = images_grp[im][:]
  curr_grid = calc_distance_grid(descriptor, renderings_grp)
  curr_grid_np = numpy.array(curr_grid)
  data_out_f[im] = curr_grid_np

# Close data files
renderings_f.close()
images_f.close()
data_out_f.close()
