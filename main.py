import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse

from model import DetNochange


def get_args():
    parser = argparse.ArgumentParser(description='<<< Train DetNochange >>>')

    parser.add_argument('--gpu_id', type=str, default='0', help='Device # for DetNochange')

    parser.add_argument('--image_size', type=int, default=512, help='Input image resolution')
    parser.add_argument('--num_channel', type=int, default=3, help='# of input image channel')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data', type=str, default=None, help='Path where the train or test data is stored')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--initial_epoch', type=int, default=0, help='Epoch at which to start training')
    parser.add_argument('--epochs', type=int, default=200, help='# of epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--decay', type=float, default=1e-6, help='Decay')
    parser.add_argument('--logdir', type=str, default=None, help='Path where the checkpoint and log will be stored')
    parser.add_argument('--weight', type=str, default=None, help='Path for model weight')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if not args.data:
        print ("ERROR: need a path for input data (--data)")
        sys.exit()

    if args.mode=='train' and not args.logdir:
        print ("ERROR: need a path for checkpoint and log (--logdir)")
        sys.exit()

    if args.mode=='test' and not args.weight:
        print ("ERROR: need a path for model weight (--weight)")
        sys.exit()
    
    #################################################################################################
    
    os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    #################################################################################################

    print (f"Device: gpu #{os.environ['CUDA_VISIBLE_DEVICES']}")

    model = DetNochange(args)

    if args.mode == 'train':
        model.train()

    elif args.mode == 'test':
        model.test()
