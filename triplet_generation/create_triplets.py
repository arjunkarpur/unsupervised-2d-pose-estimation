
# Input vars
positives_rendered_fp = "./out/positives_rendered.txt"
negatives_rendered_fp = "./out/negatives_rendered.txt"
out_fp = "./out/triplets.txt"

# Read lines from input
pos_rend_f = open(positives_rendered_fp, 'r')
neg_rend_f = open(negatives_rendered_fp, 'r')
lines_pos_rend = pos_rend_f.readlines()
lines_neg_rend = neg_rend_f.readlines()
pos_rend_f.close()
neg_rend_f.close()

# Organize inputs
pos_rend = {}
neg_rend = {}
num_pos = -1
num_neg = -1
for l in lines_pos_rend:
  split = (l.split("\n")[0]).split(",")
  pos_rend[split[0]] = split[1:]
  num_pos= len(pos_rend[split[0]])
for l in lines_neg_rend:
  split = (l.split("\n")[0]).split(",")
  neg_rend[split[0]] = split[1:]
  num_neg = len(neg_rend[split[0]])
negs_per_pos = num_neg / num_pos

# Open output file
out_f = open(out_fp, 'w')

# Print info about triplets
key = (pos_rend.keys())[0]
print "Num images: %i" % len(pos_rend)
print "Num positives per image: %i" % len(pos_rend[key])
print "Num negatives per image: %i" % len(neg_rend[key])
print "Generating %i triplets (all positives, %i negatives per positive)" % \
  ((len(pos_rend)*len(pos_rend[key])*negs_per_pos), negs_per_pos)

# Create triplets
for im in pos_rend:
  anchor = im
  positives = pos_rend[im]
  negatives = neg_rend[im]
  for i in range(len(positives)):
    curr_pos = positives[i]
    for j in range(negs_per_pos):
      curr_neg = negatives[(i*negs_per_pos)+j]
      triplet = (anchor, curr_pos, curr_neg)
      out_f.write("%s,%s,%s\n" % (triplet[0], triplet[1], triplet[2]))

# Close output file
out_f.close()
