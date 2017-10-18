import argparse
import os
from os.path import join
import sys

from utils import get_last_checkpoint

def parse_args_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suf', help='Checkpoints dir suffix.', default='')
    parser.add_argument('--pred_suf', help='prediction dir suffix.', default='')
    parser.add_argument('--model', default=None, help='Network model type.', required=True,
                        choices=['vgg16', 'vgg19', 'inception_v3', 'resnet50', 'resnet_v2_50',
                                 'resnet_v2_101', 'resnet_v2_152', 'inception_resnet_v2'])
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number [0-4]')

    parser.add_argument('--tile_size', type=int, default=224,
                        help='part of the validation split')

    parser.add_argument('--n_tta', type=int, default=12,
                        help='Number of test time augmentations')

    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None,
                        help='checkpoint name')

    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='batch_size')

    # parser.add_argument('-ng', '--num_gpus', type=int, default=1,
    #                     help='number of GPUs for parallel prediction')

    args = parser.parse_args(sys.argv[1:])
    return args


def get_pathes_predict(args, split=None):
    assert split in ['val', 'test', 'all_train']
    dir_name = args.model + args.suf
    model_dir = '/export/home/asanakoy/workspace/kaggle/amazon/checkpoints/' + dir_name
    pred_out_dir = '/export/home/asanakoy/workspace/kaggle/amazon/preds_out/' + dir_name
    if not os.path.exists(pred_out_dir):
        os.makedirs(pred_out_dir)

    img_dir_name = {
        'val': 'train',
        'test': 'test',
        'all_train': 'train'
    }
    images_dir = '/export/home/asanakoy/workspace/kaggle/amazon/input/{}-jpg/'.format(
        img_dir_name[split])
    if args.checkpoint is None:
        checkpoint_path, epoch_num = get_last_checkpoint(model_dir)
    else:
        checkpoint_path = join(model_dir, args.checkpoint)
    args.checkpoint = os.path.basename(os.path.splitext(checkpoint_path)[0])

    probs_path = join(pred_out_dir,
                      split + '_probs_{}_tta{}.npy'.format(args.checkpoint, args.n_tta))
    print 'Using checkpoint:', checkpoint_path

    return images_dir, model_dir, pred_out_dir, probs_path, checkpoint_path