
import h5py
import numpy

real_data_fp = "/mnt/localscratch/arjun/temp/image_features/alexnet/real_images.hdf5"
rend_data_fp = "/mnt/localscratch/arjun/temp/image_features/alexnet/rendered_images.hdf5"
training_triplets_fp = "../triplet_generation/out/triplets_shuffle.txt"
margin = 0.2

# Load hdf5 files
real_data_f = h5py.File(real_data_fp, 'r')
rend_data_f = h5py.File(rend_data_fp, 'r')
real_data_grp = real_data_f["imagefeatures"]
rend_data_grp = rend_data_f["imagefeatures"]

# Load triplet training txt file
training_triplets_f = open(training_triplets_fp, 'r')
triplet_lines = training_triplets_f.readlines()
training_triplets_f.close()

# Check each for error
error = []
count = 0
for l in triplet_lines:
  count += 1
  #print "Checking %i / %i" % (count, len(triplet_lines))
  split = (l.split("\n")[0]).split(",")
  anchor_str, pos_str, neg_str = \
    split[0], split[1], split[2]

  anchor_v = real_data_grp[anchor_str][:]
  pos_str_split = pos_str.split("_")
  pos_model = "_".join(pos_str_split[1:])
  pos_pose = pos_str_split[0][4:] + ".png"
  neg_str_split = neg_str.split("_")
  neg_model = "_".join(neg_str_split[1:])
  neg_pose = neg_str_split[0][4:] + ".png"
  pos_v = rend_data_grp[pos_model][pos_pose][:]
  neg_v = rend_data_grp[neg_model][neg_pose][:]

  ap = (anchor_v-pos_v).dot(anchor_v-pos_v)
  an = (anchor_v-neg_v).dot(anchor_v-neg_v)
  curr_loss = ap - an + margin
  if curr_loss > 0:
    error.append((l, curr_loss))

print "Errors (%i / %i):" % (len(error), len(triplet_lines))
for e in error:
  print e

real_data_f.close()
rend_data_f.close()

