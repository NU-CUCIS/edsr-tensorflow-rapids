# Sunwoo Lee
# <sunwoolee1.2014@u.northwestern.edu>
# 12/19/2018
#
# Data loading module for EDSR training.
########################################
import os
import random
from random import shuffle
from scipy import misc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
nproc = comm.size

class dataset():
    def __init__(self, path, image_depth, num_rows, num_cols, cropped_size, batch_size):
        self.path = path
        self.image_depth = image_depth
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cropped_size = cropped_size
        self.batch_size = batch_size
        print ("Dataset path: " + path)

        # check if a training file list has already been generated.
        # If not, generate train.txt in 'path'.
        list_path = path + "/train/list.txt"
        if os.path.exists(list_path) == False :
            print path + " does not exist. Generating training list file."
            train_list = open(list_path, "w")
            train_files = os.listdir(path + "/train/HR")
            train_list.write("\n".join(train_files))
            train_list.close()
        train_list = open(list_path, "r")
        self.train_files = train_list.readlines()
        for i in range(len(self.train_files)) :
            self.train_files[i] = self.train_files[i].rstrip('\n')
        self.num_train_images = len(self.train_files)
        train_list.close()

        # check if a testing file list has already been generated.
        # If not, generate test.txt in 'path'.
        list_path = path + "/test/list.txt"
        if os.path.exists(list_path) == False :
            print path + " does not exist. Generating test list file."
            test_list = open(list_path, "w")
            test_files = os.listdir(path + "/test/HR")
            test_list.write("\n".join(test_files))
            test_list.close()
        test_list = open(list_path, "r")
        self.test_files = test_list.readlines()
        for i in range(len(self.test_files)) :
            self.test_files[i] = self.test_files[i].rstrip('\n')
        self.num_test_images = len(self.test_files)
        test_list.close()

        # calculate the number of iterations per epoch
        self.num_train_iterations = len(self.train_files) / batch_size

        # when testing, calculate PSNR one image after another
        self.num_test_iterations = len(self.test_files)

    def shuffle(self):
        shuffle(self.train_files)
        # rank 0 broadcasts this random order so that
        # all ranks have a consistent view.
        self.train_files = comm.bcast(self.train_files, root = 0)

    def crop_image(self, hr_image, lr_image):
        h_off = random.sample(range(self.num_rows - self.cropped_size + 1), 1)[0]
        w_off = random.sample(range(self.num_cols - self.cropped_size + 1), 1)[0]
        return hr_image[h_off : h_off + self.cropped_size, w_off : w_off + self.cropped_size], \
               lr_image[h_off : h_off + self.cropped_size, w_off : w_off + self.cropped_size]

    def get_train_batch(self, index, size):
        files = self.train_files[index : index + size]
        hr_images = []
        lr_images = []
        for i in range(size) :
            # read both high- and low-resolution images.
            hr_image = misc.imread(self.path + "/train/HR/" + files[i], 'RGB')
            lr_image = misc.imread(self.path + "/train/LR/" + files[i], 'RGB')

            # crop a pair of images with the same random offsets
            hr_image, lr_image = self.crop_image(hr_image, lr_image)

            # reshape them and append to the list
            hr_image = hr_image.reshape(self.cropped_size, self.cropped_size, self.image_depth)
            lr_image = lr_image.reshape(self.cropped_size, self.cropped_size, self.image_depth)
            hr_images.append(hr_image)
            lr_images.append(lr_image)
        return lr_images, hr_images

    def get_test_batch(self, index, size):
        files = self.test_files[index : index + size]
        hr_images = []
        lr_images = []
        for i in range(size) :
            # read both high- and low-resolution images.
            hr_image = misc.imread(self.path + "/test/HR/" + files[i], 'RGB')
            lr_image = misc.imread(self.path + "/test/LR/" + files[i], 'RGB')

            # reshape them and append to the list
            hr_image = hr_image.reshape(self.num_rows, self.num_cols, self.image_depth)
            lr_image = lr_image.reshape(self.num_rows, self.num_cols, self.image_depth)
            hr_images.append(hr_image)
            lr_images.append(lr_image)
        return lr_images, hr_images
