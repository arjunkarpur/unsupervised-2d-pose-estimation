
import h5py
import numpy as np

real_data_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/feature_data/finetune/triplets_one/1_28mil/margin_0_2/real_images.hdf5"
rend_data_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/feature_data/finetune/triplets_one/1_28mil/margin_0_2/rendered_images.hdf5"
training_triplets_fp = "../triplet_generation/out/test_set/triplets_shuffle.txt"
margin = 0.2
l2_norm = False

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

  if l2_norm:
    anchor_v = anchor_v / (np.sqrt(anchor_v.dot(anchor_v)))
    pos_v = pos_v / (np.sqrt(pos_v.dot(pos_v)))
    neg_v = neg_v / (np.sqrt(neg_v.dot(neg_v)))

  ap = (anchor_v-pos_v).dot(anchor_v-pos_v)
  an = (anchor_v-neg_v).dot(anchor_v-neg_v)
  curr_loss = ap - an + margin
  if curr_loss > 0:
    error.append((l, curr_loss))

# Print data
total_loss = 0
for e in error:
  total_loss += e[1]
avg_loss_per_error = float(total_loss)/float(len(error))
avg_loss_all_triplets = float(total_loss)/float(len(triplet_lines))
error_rate = float(100)*float(len(error))/float(len(triplet_lines))
print "Error rate (%i / %i): %f %%" % (len(error), len(triplet_lines), error_rate)
print "Average loss per error: %f" % (avg_loss_per_error)
print "Average loss (all triplets): %f" % (avg_loss_all_triplets)

real_data_f.close()
rend_data_f.close()
