import h5py
import subprocess 

# Inputs vars
src_hdf5_fps = []
dest_hdf5_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/dist_grids/finetune/finetune__dist_grids.hdf5"
for i in range(8):
  src_path = \
    "/mnt/localscratch/arjun/temp/image_features/finetune/grids/finetune_dist_grid_%i.hdf5"%i
  src_hdf5_fps.append(src_path)

# Create output hdf5
dest_hdf5_f = h5py.File(dest_hdf5_fp, 'w')
dest_hdf5_f.close()

# Add data from each input file to out
file_count = 0
for src_fp in src_hdf5_fps:
  file_count += 1
  print "Joining %s: %i/%i" % (src_fp, file_count, len(src_hdf5_fps))
  src_f = h5py.File(src_fp, 'r')
  keys = src_f.keys()
  src_f.close()
  for key in keys:
    cmd = "h5copy -i %s -o %s -s %s -d %s" % \
      (src_fp, dest_hdf5_fp, key, key)
    subprocess.call(cmd.split(" "))
