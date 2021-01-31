# P06

Image Segmentation application project for medical images using Tensorflow 2.0.
This is part of my solution (benchmark) for one of the tasks of the challenge [QUBIQ](https://qubiq.grand-challenge.org/).

### Objective
The purpose of the challenge is to benchmark segmentation algorithms returning uncertainty estimates (probability scores, variability regions, etc.) of structures in medical imaging segmentation tasks. 

### Dataset
Training and test data comprised 7 binary segmentation tasks in four different CT and MR data sets.
For this repo, I've just shown my solution for Brain growth images (MRI)'s task, which has 39 cases, 7 expert's annotations on its training set. Validation set has 5 cases and was available just for participants.

### Solution
For my benchmark I used U-Net as it is a very known architecture for Image Segmentation on medical images. Focal loss also was used as it gave better results than classic loss fuctions.

### Results
As can be seen, the results were very close to the ground truth on validation set.

![Alt text](imgs/results.PNG?raw=true "Results on Validation set")