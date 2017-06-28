
gt_fp = "./gt.txt"
poses_fp = "../out/finetune_poses.txt"

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

def calculate_error(gt_pose, predicted_pose):
  gt_ang_1 = int(gt_pose) % 30
  gt_ang_2 = int(gt_pose) / 30
  predicted_ang_1 = int(predicted_pose) % 30
  predicted_ang_2 = int(predicted_pose) / 30

  ind_1_diff = \
    max(gt_ang_1 - predicted_ang_1, predicted_ang_1 - gt_ang_1)
  ind_2_diff = \
    max(gt_ang_2 - predicted_ang_2, predicted_ang_2 - gt_ang_2)

  ang_1_diff = min(ind_1_diff*12, 360-(ind_1_diff*12))
  ang_2_diff = min(ind_2_diff*30, 360-(ind_2_diff*30))
  #print ang_1_diff, ang_2_diff
  return (ang_1_diff, ang_2_diff)

# Test
total = 1
total_one_error = 0
total_two_error = 0
print "Testing with %i ground truth images/poses" % len(gt)
for l in poses_lines:
  split = (l.strip()).split(" ")
  if split[0] not in gt:
    continue
  
  gt_pose = gt[split[0]]
  estimate = int(split[1])
  (one_error, two_error) = \
    calculate_error(gt_pose, estimate)

  total_one_error += one_error
  total_two_error += two_error
  total += 1

avg_one_error = float(float(total_one_error)/float(total))
avg_two_error = float(float(total_two_error)/float(total))
print "Angle 1 (azimuth) error: %f degrees" % avg_one_error
print "Angle 2 (elevation) error: %f degrees" % avg_two_error
