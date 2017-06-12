
gt_fp = "./gt.txt"
poses_fp = "../alexnet_fc7_poses.txt"
relax = 5

# Read GT lines
gt_f = open(gt_fp, 'r')
gt_lines = gt_f.readlines()
gt_f.close()

# Create GT
gt = {}
for l in gt_lines:
  split = (l.strip()).split(" ")
  gt[split[0]] = int(split[1])

# Read poses lines
poses_f = open(poses_fp, 'r')
poses_lines = poses_f.readlines()
poses_f.close()

# Test
total = 0
positive = 0
for l in poses_lines:
  split = (l.strip()).split(" ")
  if split[0] not in gt:
    continue
  
  gt_pose = gt[split[0]]
  estimate = int(split[1])

  total += 1
  if (estimate <= gt_pose + relax) and \
     (estimate >= gt_pose - relax):
    positive += 1

accuracy = float(float(positive)/float(total))
print "%i / %i positive\t %f" % \
  (positive, total, accuracy)
