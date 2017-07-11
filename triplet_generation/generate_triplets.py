import sys
import h5py
import random

# Inputs vars
dist_grids_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/dist_grids/alexnet_fc7_dist_grids.hdf5"
poses_fp = "../pose_estimation/out/split/%i.txt" % int(sys.argv[1])
out_fp = "./out/triplets_four/split/triplets_%i.txt" % int(sys.argv[1])
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
pos_bin_probs = [.60, .20, .10, .07, .03]
neg_bin_probs = [.03, .07, .10, .20, .60]
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

  # Create bins
  bin_sizes = []
  for i in range(num_bins):
    if i == (num_bins-1):
      size = len(sorted_dists) / num_bins
    else:
      size = (len(sorted_dists) / num_bins) + (len(sorted_dists) % num_bins)
    bin_sizes.append(size)

  """
  # BIN INFO PRINTING
  for b in range(len(bin_sizes)):
    print "BIN %i" % b
    print "start: ", sorted_dists[b*bin_sizes[0]]
    print "end: ", sorted_dists[b*bin_sizes[0] + bin_sizes[b] - 1]
  """

  # Generate triplets
  for i in range(num_positives):
    anchor = name
    # Pick 1 positive and #NEGS_PER_POS negatives
    pos_bin = pos_bin_bag[random.randint(0,99)]
    bin_size = bin_sizes[pos_bin]
    bin_ind = random.randint(0,bin_size-1)
    positive = sorted_dists[pos_bin*bin_sizes[0] + bin_ind]
    pos_name = "rend%i_%s" % (positive[1], str(positive[0]))

    # Pick #NEGS_PER_POS negatives
    for j in range(negs_per_pos):
      neg_bin = neg_bin_bag[random.randint(0,99)]
      bin_size = bin_sizes[neg_bin]
      bin_ind = random.randint(0, bin_size-1)
      negative = sorted_dists[neg_bin*bin_sizes[0] + bin_ind]

      neg_name = "rend%i_%s" % (negative[1], str(negative[0]))
      triplet = "%s,%s,%s" % (anchor, pos_name, neg_name)
      out_f.write("%s\n" % triplet)

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
