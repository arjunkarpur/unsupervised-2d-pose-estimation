import h5py
import random

# Inputs vars
dist_grids_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/dist_grids/alexnet_fc7_dist_grids.hdf5"
poses_fp = "../pose_estimation/out/alexnet_fc7_poses.txt"
out_fp = "./out/negatives_rendered.txt"
top_n = 5
num_negatives = 125

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

def calculate_angle_diff(a, b):
  a_ang_1 = int(a) % 30
  a_ang_2 = int(a) / 30
  b_ang_1 = int(b) % 30
  b_ang_2 = int(b) / 30

  ind_1_diff = \
    max(a_ang_1 - b_ang_1, b_ang_1 - a_ang_1)
  ind_2_diff = \
    max(a_ang_2 - b_ang_2, b_ang_2 - a_ang_2)

  ang_1_diff = min(ind_1_diff*12, 360-(ind_1_diff*12))
  ang_2_diff = min(ind_2_diff*30, 360-(ind_2_diff*30))
  return (ang_1_diff, ang_2_diff) 

# Generate NUM_NEGATIVES per real image
count = 0
for im in im_poses:
  count += 1
  print "Generating negatives: %i / %i" % (count, len(im_poses))
  name, pose = im[0], int(im[1])
  curr_dist_grid = dist_grids_f[name]

  # Enumerate all potential negatives for each image
  negatives = []
  for curr_p in range(len(curr_dist_grid)):
    add = []
    """
    if pose == curr_p:
      add = curr_dist_grid[curr_p][top_n:]
    else:
      add = curr_dist_grid[curr_p][:]
    """
    diff_1, diff_2 = calculate_angle_diff(pose, curr_p)
    if diff_1 > 50 or diff_2 > 50:
        add = curr_dist_grid[curr_p][:]
    for l in add:
      negatives.append("pose%i_%s" % (curr_p,str(l[0])))

  # Select NUM_NEGATIVES
  chosen_negatives = random.sample(negatives, num_negatives)

  # Write to file
  out_line = "%s" % name
  for p in chosen_negatives:
    out_line += ",%s" % str(p)
  out_f.write("%s\n" % out_line)

# Close files
dist_grids_f.close()
out_f.close()
