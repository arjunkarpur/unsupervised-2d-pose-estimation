import h5py

# Inputs vars
top_n = 5

#dist_grid_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/dist_grids/alexnet_fc7_dist_grids.hdf5"
dist_grid_fp = "/mnt/localscratch/arjun/temp/dist_grids/alexnet_fc7_dist_grids.hdf5"
#dist_grid_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/dist_grids/finetune/finetune_dist_grids.hdf5"

out_fp = "./out/alexnet_fc7_poses.txt"
#out_fp = "./out/finetune_poses.txt"

#imlist_fp = "./test_error/test_set.txt"
imlist_fp = "../inputs/real_images.txt"

# Open distance grid data file for reading
dist_grid_f = h5py.File(dist_grid_fp, 'r')

# Read list of files to perform pose estimation
im_list = []
imlist_f = open(imlist_fp, 'r')
lines = imlist_f.readlines()
imlist_f.close()
for l in lines:
  im_list.append(l.split("\n")[0])

# Open output file
out_f = open(out_fp, 'w')

####################
#  BEGIN FUNCTIONS  #
#####################

def pose_estimation(dist_grid):
  min_pose = -1
  min_sum = -1
  for p in range(len(dist_grid)):
    # Calculate sum of L2 distances for top N renderings per pose
    curr_sum = 0
    for n in range(top_n):
      curr_sum += float(dist_grid[p][n][1])

    # Set the best pose as one with min top_n sum
    if (min_pose == -1 and min_sum == -1) or \
       (curr_sum < min_sum):
      min_pose = p
      min_sum = curr_sum

  print "Avg distance: %f" % (curr_sum/top_n)
  return min_pose

#####################
#   END FUNCTIONS   #
#####################

print "Performing pose estimation for %i images" % len(im_list)
count = 0
for im in im_list:
  count += 1
  print "Pose estimation: %i / %i" % (count, len(im_list))
  if im not in dist_grid_f:
    print "ERROR: No distance grid available for image %s" % im
    continue
  curr_pose = pose_estimation(dist_grid_f[im])
  out_f.write("%s %s\n" % (im, str(curr_pose)))

# Close files
dist_grid_f.close()
out_f.close()
