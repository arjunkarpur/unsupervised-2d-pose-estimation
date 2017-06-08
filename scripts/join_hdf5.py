import h5py

# Inputs vars
src_hdf5_fps = []
for i in range(8):
  src_path = \
    "/net/cvcfs/storage/skull-atlas/imgscrape/dist_grids/alexnet_fc7_dist_grid_%i.hdf5"%i
  src_hdf5_fps.append(src_path)
dest_hdf5_fp = "/net/cvcfs/storage/skull-atlas/imgscrape/dist_grids/alexnet_fc7_dist_grids.hdf5"

# Create output hdf5
dest_hdf5_f = h5py.File(dest_hdf5_fp, 'w')

# Add data from each input file to out
for src_fp in src_hdf5_fps:
  src_f = h5py.File(src_fp, 'r')
  for dset in src_f:
    dest_hdf5_f[dset] = src_f[dset]
  src_f.close()

# Close output hdf5
dest_hdf5_f.close()
