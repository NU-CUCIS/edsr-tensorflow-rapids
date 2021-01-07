import data
import model
import argparse
from data import dataset
from model import edsr_model

parser = argparse.ArgumentParser()
#parser.add_argument("--dataset", default = "/home/slz839/dataset/phantom_v2")
#parser.add_argument("--num_rows", default = 256, type = int)
#parser.add_argument("--num_cols", default = 256, type = int)
parser.add_argument("--dataset", default = "/home/slz839/dataset/simulation")
parser.add_argument("--num_rows", default = 312, type = int)
parser.add_argument("--num_cols", default = 640, type = int)
parser.add_argument("--imgdepth", default = 1, type = int)
parser.add_argument("--cropsize", default = 32, type = int)
parser.add_argument("--batchsize", default = 16, type = int)
parser.add_argument("--layers", default = 16, type = int)
parser.add_argument("--filters", default = 256, type = int)
parser.add_argument("--epochs", default = 1, type = int)
parser.add_argument("--resume", default = 0, type = int)
parser.add_argument("--test", default = True, type = bool)

args = parser.parse_args()

# Initialize dataset.
training_data = dataset(args.dataset,
                        args.imgdepth,
                        args.num_rows,
                        args.num_cols,
                        args.cropsize,
                        args.batchsize)

edsr = edsr_model(training_data.num_train_iterations,
                  training_data.num_test_iterations,
                  args.batchsize,
                  args.layers,
                  args.filters,
                  args.imgdepth,
                  args.cropsize,
                  args.test)

edsr.set_functions(training_data.get_train_batch,
                   training_data.get_test_batch,
                   training_data.shuffle)

edsr.train(args.epochs, args.resume)
