import h5py
import random

# Inputs vars
dist_grids_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/dist_grids/alexnet_fc7_dist_grids.hdf5"
poses_fp = "../pose_estimation/out/alexnet_fc7_poses.txt"
out_fp = "./out/triplets.txt"
num_positives = 50
negs_per_pos = 4

# Open dist grid file
dist_grids_f = h5py.File(dist_grids_fp, 'r')

# Read image names w/ poses
im_poses = []
poses_f = open(poses_fp, 'r')
lines = poses_f.readlines()
poses_f.close()
for l in lines:
  l = l.split("\n")[0]
  split = l.split(" ")
  im_poses.append((split[0], int(split[1])))

# Create output file
out_f = open(out_fp, 'w')

# Bins for selecting triplets
num_bins = 5
pos_bin_probs = [.50, .20, .15, .10, .05]
neg_bin_probs = [.05, .10, .15, .20, .50]
pos_bin_bag = []
neg_bin_bag = []
for i in range(len(pos_bin_probs)):
  for j in range(int(100*pos_bin_probs[i])):
    pos_bin_bag.append(i)
  for j in range(int(100*neg_bin_probs[i])):
    neg_bin_bag.append(i)
random.shuffle(pos_bin_bag)
random.shuffle(neg_bin_bag)

# Generate N positives per real image
count = 0
for im in im_poses:
  count += 1
  print "Generating positives: %i / %i" % (count, len(im_poses))
  name, pose = im[0], int(im[1])

  all_dists = []
  for i in xrange(len(dist_grids_f[name])):
    for j in xrange(len(dist_grids_f[name][i])):
      curr_pose = i
      curr_name = dist_grids_f[name][i][j][0]
      curr_dist = dist_grids_f[name][i][j][1]
      all_dists.append((curr_name, curr_pose, curr_dist))
  sorted_dists = sorted(all_dists, key=lambda x: float(x[2]))

  for i in range(num_positives):
    pass
    #TODO: pick positive
    #TODO: pick #NEGS_PER_POS negatives
    #TODO: write triplets

  """
  positives = dist_grids_f[name][pose][:top_n]
  out_line = "%s" % name
  for p in positives:
    p_mod = "rend%i_%s" % (pose, str(p[0]))
    out_line += ",%s" % p_mod
  out_f.write("%s\n" % out_line)
  """

# Close files
dist_grids_f.close()
out_f.close()
