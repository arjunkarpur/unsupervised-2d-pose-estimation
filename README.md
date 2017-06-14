# Unsupervised 2D Pose Estimation

Unsupervised method of performing pose estimation of 2D images using renderings of 3D object models.

**Pipeline:**
  1) Use network to determine features for real & rendered images (start w/ AlexNet trained on ImageNet)
  2) Calculate distance grid between real images and rendered images (dim: #poses x #models)
  3) Perform pose estimation 
  4) Generate triplets 
  5) Fine tune same network using triplets (& save snapshot of network)
  6) Perform pose estimation testing for error rates
  7) GOTO 1...?

**To-do (w/ priority):**  
  * [1] Write network fine tuning code  
  * [1] Change triplet code to dynamically find triplets during training (100 hardest)
  * [1] Determine test dataset for pose estimation task  
  * [2] Verify Steps 1-4 are correct
  * [2] Add in triplet generation using real-to-real comparisons (pos and neg)
  * [2] Bring in code for Step 1 (right now not in repo)
  * [3] Create script to automate pipeline  
  * [3] Change distance grid computation code to work with UTCS Condor for faster runtime
  * [3] Add detailed written description of project to README
