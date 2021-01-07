# Sunwoo Lee
# <sunwoolee1.2014@u.northwestern.edu>
# 12/19/2018
#
# Model setting module for EDSR training.
#########################################
import os
import time
import math
import numpy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import horovod.tensorflow as hvd
from tqdm import tqdm

def calc_psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if(mse == 0):
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))

def res_block(x, num_filters = 64, filter_size = [3, 3], scale = 0.1):
    layer = slim.conv2d(x, num_filters, filter_size, activation_fn = None)
    layer = tf.nn.relu(layer)
    layer = slim.conv2d(layer, num_filters, filter_size, activation_fn = None)
    layer *= scale
    return x + layer

class edsr_model():
    def __init__(self, num_train_iterations, num_test_iterations, batch_size, \
                 num_layers = 16, num_filters = 256, input_depth = 1, input_size = 32, do_test = False):
        print ("Initializing EDSR model with " + \
                str(num_layers) + " layers, each of " +\
                str(num_filters) + " filters.")
        print ("Initializing horovod...")
        hvd.init()

        self.num_train_iterations = num_train_iterations
        self.num_test_iterations = num_test_iterations
        self.do_test = do_test
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.input = x = tf.placeholder(tf.float32, [None, None, None, input_depth])
        self.target = y = tf.placeholder(tf.float32, [None, None, None, input_depth])
        self.lr = tf.placeholder(tf.float32, [])

        # Define EDSR model architecture.
        images = x
        x = slim.conv2d(images, num_filters, [3, 3])
        conv_1 = x

        for i in range(num_layers):
            x = res_block(x, num_filters)

        x = slim.conv2d(x, num_filters, [3, 3])
        x += conv_1
        output = slim.conv2d(x, input_depth, [3, 3])

        # self.out will be used to make a prediciton
        self.out = tf.clip_by_value(output, 0.0, 255.0)

        # Define loss function.
        self.loss = loss = tf.reduce_mean(tf.losses.absolute_difference(y, output))

        # Settings for using multiple GPUs in each node.
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
        print ("Done setting EDSR model!")

    def set_functions(self, get_train_batch_fn, get_test_batch_fn, data_shuffle_fn):
        self.get_train_batch_fn = get_train_batch_fn
        self.get_test_batch_fn = get_test_batch_fn
        self.data_shuffle_fn = data_shuffle_fn

    def checkpoint(self, step, checkpoint_path = "checkpoints"):
        if os.path.isdir(checkpoint_path) == False:
            os.mkdir(checkpoint_path)
        newpath = checkpoint_path + "/" + str(step)
        if os.path.isdir(newpath) == False:
            os.mkdir(newpath)
        self.saver.save(self.sess, newpath + "/edsr-" + str(step))
        print ("Done checkpointing.")

    def resume(self, step, checkpoint_path = "checkpoints"):
        path = checkpoint_path + "/" + str(step)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
        print ("Done reading model parameters from " + path)

    def train(self, epochs = 10, resume_from = 0, checkpoint_path = "checkpoints"):
        # Define optimizer.
        lr = 1e-4
        opt = tf.train.AdamOptimizer(self.lr)
        distributed_opt = hvd.DistributedOptimizer(opt)
        train_op = distributed_opt.minimize(self.loss)

        # Initialize parameters.
        self.sess.run(tf.global_variables_initializer())

        if resume_from > 0:
            print ("Resuming training...")
            self.resume(resume_from, checkpoint_path)
        else:
            print ("Starting training from the scratch...")

        for i in range(epochs):
            # Shuffle the data each epoch.
            self.data_shuffle_fn()

            index = 0
            local_batch_size = self.batch_size / hvd.size()
            for j in tqdm(range(self.num_train_iterations)):
                x, y = self.get_train_batch_fn(index + (hvd.rank() * local_batch_size), local_batch_size)
                index += self.batch_size

                feed = {
                    self.input:x,
                    self.target:y,
                    self.lr:lr
                }
                self.sess.run(train_op, feed)

            if (i + 1) % 10 == 0 and i > 0:
                # When validating the model, just use rank 0
                if hvd.rank() == 0:
                    self.checkpoint(resume_from + i + 1, checkpoint_path)
                    if self.do_test:
                        self.test(resume_from + i + 1, checkpoint_path)

    def evaluate(self, x):
        output = self.sess.run(self.out, feed_dict = {self.input:x})
        return output

    def test(self, resume_from, checkpoint_path = "checkpoints"):
        # Read the parameter values from the checkpoint.
        self.resume(resume_from, checkpoint_path)

        # Evaluate the test dataset.
        average = 0
        for i in tqdm(range(self.num_test_iterations)):
            x, y = self.get_test_batch_fn(i, 1)
            output = self.evaluate(x)
            psnr = calc_psnr(output, y)
            average += psnr
        average /= self.num_test_iterations
        print ("Average PSNR: " + str(average))
        f = open('acc.txt', 'a')
        f.write(str(average) + '\n')
        f.close()
