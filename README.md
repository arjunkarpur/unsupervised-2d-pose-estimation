# Unsupervised 2D Pose Estimation

**TODO:** Written description of project

**Pipeline:**
  1) Use network to determine features for real & rendered images (start w/ AlexNet trained on ImageNet)
  2) Calculate distance grid between real images and rendered images (dim: #poses x #models)
  3) Perform pose estimation 
  4) Generate triplets 
  5) Fine tune same network using triplets (& save snapshot of network)
  6) Perform pose estimation testing for error rates
  7) GOTO 1...

**TODO (Priority 1 (high) through 3 (low):**
  [1] Write triplet generation code
  [1] Write network fine tuning code
  [1] Determine test dataset for pose estimation task
  [1] Write pose estimation testing code
  [2] Bring in code for Step 1 (right now not in repo)
  [3] Create script to automate pipeline
