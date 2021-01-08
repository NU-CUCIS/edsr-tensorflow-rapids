# Parallel EDSR-TensorFlow
TensorFlow (1.15) + Horovod implementation of image restoration using Enhanced
Deep Super-Resolution (EDSR), a deep residual network proposed in [1].  The
data input part was implemented to support the Climate dataset.  The training
is parallelized based on data parallelism such that each mini-batch is evenly
distributed to all MPI processes and independently processed.  Then, the local
gradients are averaged among all the processes using inter-process
communications in Horovod.  When studying the scaling performance in our paper,
we used a large-batch size (256), and adjusted the learning rate based on the
linear scaling rule [2].

[1]: Lim et al., Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPR workshop 17
[2]: Goyal et al., Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, Arxiv 18

## Run instructions
Start parallel training using MPI (the input options are shown below).
The model parameters are checkpointed at the end of every epoch.
```
mpiexec -n 16 python train.py --batchsize=16 --epochs=1000
```

### Command-line options for training
* `--dataset`: the path to the top directory of the dataset, default = "/home/slz839/dataset/simulation"
* `--num_rows`: the input data row size, default = 312
* `--num_cols`: the input data column size, default = 640
* `--imgdepth`: the input image depth (number of channels), default = 1
* `--cropsize`: the crop size (a random square region is cropped), default = 32
* `--batchsize`: the mini-batch size, default = 16
* `--layers`: the number of residual blocks, default = 16
* `--filters`: the number of filters per layer, default = 256
* `--epochs`: the number of training epochs, default = 1
* `--resume`: 0: initial training, 1: resume from a checkpoint, default = 0
* `--test`: True: perform the evaluation using test dataset at the end of every epoch, default = True

---

# Climate dataset
This dataset contains atmospheric flow images for barotropic instability test
that assesses the quality of atmospheric numerical methods [1].  The test
involves perturbing the atmospheric flow by a localized bump to the balanced
height field.  The localized bump to the balanced height is set to three
different values, 120m, 180m, and 240m.

[1] Galewsky et al., An initial-value problem for testing numerical models of
the global shallow water equations (vol 56A, pg 429, 2004), Tellus A: Dynamic
Meteorology and Oceanography

## Download dataset
[climate-rapids.tar.gz](http://cucis.ece.northwestern.edu/projects/RAPIDS/climate-rapids.tar.gz)

## Dataset organization
The dataset consists of 4,000 images generated from simulation using 120m and
240m height settings, 2,000 for each.  For each setting, the first 700 images
were discarded and only 1,300 images were used in our study.  The images
generated with 180m height setting were not used for scaling performance study
and we do not keep the images in this dataset.

* `train`: 80% of the total images (2,080). The file names are contiguous IDs
  (1.png ~ 2080.png).
* `test`: 10% of the total images (260). The file names are not contiguous
  (705.png ~ 1999.png).
* `val`: 520 images. The file names are contiguous IDs (1.png ~ 520.png).

### Folder structure
* simulation: root directory
  * simulation/train: image files used for training
    * simulation/train/HR: the original images for training
    * simulation/train/LR: the JPEG-compressed images for training
    * simulation/train/list.txt: the list of the file names.
  * simulation/test: image files used for validation
    * simulation/test/HR: the original images for validation
    * simulation/test/LR: the JPEG-compressed images for validation
    * simulation/test/list.txt: the list of the file names.
  * simulation/val: image files used for hyper-parameter tuning
    * simulation/val/HR: the original images for hyper-parameter tuning
    * simulation/val/HR: the JPEG-compressed images for hyper-parameter tuning

### Sample details
Each sample is a pair of High Resolution (HR) and Low Resolution (LR) PNG
images of size 640 x 312.  The HR image is the original image while LR image is
the JPEG-compressed image with the compression rate of 0.01.

In our work, we solved a image regression problem considering the LR images as
input data and the HR images as label.  Given a batch of LR images, neural
networks estimate the corresponding HR images, and the outputs are compared
with the original HR images.

## Publication
* Sandeep Madireddy, Ji Hwan Park, Sunwoo Lee, Prasanna Balaprakash, Shinjae
  Yoo, Wei-keng Liao, Cory Hauck, M. Paul Laiu, and Richard Archibald.
  [In Situ Compression Artifact Removal in Scientific Data Using Deep Transfer Learning and Experience Replay](https://iopscience.iop.org/article/10.1088/2632-2153/abc326/meta),
  Machine Learning: Science and Technology, Volume 2, Number 2, IOP Publishing
  Ltd, December 2020.

## Developers
* Sunwoo Lee <<sunwoolee1.2014@u.northwestern.edu>>
* Wei-keng Liao <<wkliao@northwestern.edu>>

## Collaborators
* Argonne National Laboratory
  * Sandeep Madireddy <<smadireddy@anl.gov>>
  * Prasanna Balaprakash <<pbalapra@anl.gov>>
* Oak Ridge National laboratory
  * Richard Archibald <<archibaldrk@ornl.gov>>

## Questions/Comments

## Acknowledgment
This material is based upon work supported by the U.S. Department of Energy,
Office of Science, Office of Advanced Scientific Computing Research, Scientific
Discovery through Advanced Computing (SciDAC) program. This project is a joint
work of Northwestern University and Argonne National Laboratory supported by
the [RAPIDS Institute](https://rapids.lbl.gov).

