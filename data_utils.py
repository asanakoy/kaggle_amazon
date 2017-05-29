import argparse
import os
from os.path import join
import sys

ROOT_DIR = '/export/home/asanakoy/workspace/kaggle/amazon'


def parse_args_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suf', help='Checkpoints dir suffix.', default='')
    parser.add_argument('--model', default=None, help='Network model type.', required=True,
                        choices=['vgg16', 'vgg19', 'inception_v3', 'resnet_v2_50',
                                 'resnet_v2_101', 'resnet_v2_152', 'inception_resnet_v2'])
    parser.add_argument('--val_part', type=float, default=0.2,
                        help='part of the validation split')

    parser.add_argument('--tile_size', type=int, default=224,
                        help='part of the validation split')

    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None,
                        help='checkpoint name')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    args = parser.parse_args(sys.argv[1:])
    return args


def get_pathes_predict(args, split=None):
    assert split in ['val', 'test']
    dir_name = args.model + args.suf
    model_dir = '/export/home/asanakoy/workspace/kaggle/amazon/checkpoints/' + dir_name
    pred_out_dir = '/export/home/asanakoy/workspace/kaggle/amazon/preds_out/' + dir_name
    if not os.path.exists(pred_out_dir):
        os.makedirs(pred_out_dir)

    probs_path = join(pred_out_dir, split + '_probs.npy')
    img_dir_name = {
        'val': 'train',
        'test': 'test'
    }
    images_dir = '/export/home/asanakoy/workspace/kaggle/amazon/input/{}-jpg/'.format(
        img_dir_name[split])
    checkpoint_path = join(model_dir, args.checkpoint)
    print 'Using checkpoint:', checkpoint_path

    return images_dir, model_dir, pred_out_dir, probs_path, checkpoint_path