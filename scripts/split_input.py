fp = "inputs/real_images.txt"
f = open(fp, 'r')
lines = f.readlines()
f.close()

num = len(lines)
groups = 8
split = num/groups

split_lines = [[] for x in range(groups)]

count = 0
for l in lines:
  group = count/split
  if group >= groups:
    group = groups-1
  split_lines[group].append(l.split("\n")[0])
  count += 1

for i in range(groups):
  out = file("inputs/%i.txt" % i, 'w')
  for line in split_lines[i]:
    out.write("%s\n" % line)
  out.close()
