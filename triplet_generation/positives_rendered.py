import h5py

# Inputs vars
dist_grids_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/dist_grids/alexnet_fc7_dist_grids.hdf5"
poses_fp = "../pose_estimation/alexnet_fc7_poses.txt"
out_fp = "./out/positives_rendered.txt"
top_n = 5

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

# Generate N positives per real image
count = 0
for im in im_poses:
  count += 1
  print "Generating positives: %i / %i" % (count, len(im_poses))
  name, pose = im[0], int(im[1])
  positives = dist_grids_f[name][pose][:top_n]
  out_line = "%s" % name
  for p in positives:
    p_mod = "rend%i_%s" % (pose, str(p[0]))
    out_line += ", %s" % p_mod
  out_f.write("%s\n" % out_line)

# Close files
dist_grids_f.close()
out_f.close()
