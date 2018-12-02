import argparse
import numpy as np

def parse_arg():
    parser = argparse.ArgumentParser(prog='my_program')
    parser.add_argument('-train', type=str, default=False, help="path to csv file")
    parser.add_argument('-weight', type=str, default=False, help="Load model's weigth")
    parser.add_argument('-e',"--epoch", type=int, default=100, help="Epoch number")
    parser.add_argument('-batch', "--batch", type=int, default=1, help="Batch size")
    parser.add_argument('-lr',"-lr", type=float, default=1, help='Learning rate')
    parser.add_argument('-nesterov', type=float, default=0, help='Nesterov gamma factor')
    parser.add_argument('-best', "--best", action='store_true', default=False, help="save best weights")
    parser.add_argument('-net_shape', type=str, default='100 70 20', help="hidden layers")
    args = parser.parse_args()
    if not args.train:
        print("Add path to your data")
        exit(1)
    return args

def parse_shape(path):
     L = np.load(path)
     return L[2]