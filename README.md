# Unsupervised 2D Pose Estimation

**WORK IN PROGRESS**

Unsupervised method of performing pose estimation of 2D images using renderings of 3D object models. For detailed write-up with results, visit: http://www.cs.utexas.edu/~akarpur/files/Unsupervised_Skull_Writeup_Summer17.pdf

**Pipeline:**  
  1) Render images of N models in M known poses. Collect 'real images' scraped from web (not included in repo - see https://github.com/arjunkarpur/multi-view-rendering)
  2) Use network to determine features for real & rendered images (start w/ AlexNet trained on ImageNet)
  3) Calculate distance grid between real images and rendered images (dim: #poses x #models)
  4) Perform pose estimation 
  5) Generate triplets 
  6) Fine tune same network using triplets
  7) Perform pose estimation testing for error rates (repeat steps 2-4 w/ new network weights)

**To-do (w/ priority):**
  * [1] Change triplet code to dynamically find triplets during training to speed up training time
  * [2] Add in triplet generation using real-to-real comparisons (pos and neg)
  * [2] Change distance grid computation code to work with UTCS Condor for faster runtime
  * [3] Add in commands and detailed instructions on how to run in README
  * [3] Create script to automate pipeline
