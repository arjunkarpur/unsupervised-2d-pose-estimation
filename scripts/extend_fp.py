
import os
in_fp = "../inputs/real_images.txt"
dir_base = "/mnt/localscratch/arjun/real_image_dump"
out_fp = "../inputs/real_images_full.txt"

in_f = open(in_fp, 'r')
lines = in_f.readlines()
in_f.close()

out_f = open(out_fp, 'w')
for l in lines:
  out_f.write("%s\n" % os.path.join(dir_base, l.split("\n")[0]))
out_f.close()
