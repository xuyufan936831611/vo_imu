#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import cv2
import argparse
from evaluation_utils import *
from glob import glob
import matplotlib.pyplot as plt
import scipy.misc

parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
parser.add_argument('--split', type=str, help='data split, kitti or eigen', required=True)
parser.add_argument('--predicted_disp_path', type=str, help='path to estimated disparities', required=True)
parser.add_argument('--gt_path', type=str, help='path to ground truth disparities', required=True)
parser.add_argument('--min_depth', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth', type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--output_dir', type=str)
parser.add_argument('--output_dir_gt', type=str)
parser.add_argument('--img_height',type=int, default=375)
parser.add_argument('--img_width', type=int,default=1242)
args = parser.parse_args()

CMAP = 'plasma'
def _gray2rgb(im, cmap=CMAP):
  cmap = plt.get_cmap(cmap)
  rgba_img = cmap(im.astype(np.float32))
  rgb_img = rgba_img[:,:,:3]
  return rgb_img


def _normalize_depth_for_display(depth,
                                 pc=95,
                                 crop_percent=0,
                                 normalizer=None,
                                 cmap=CMAP):
  """Converts a depth map to an RGB image."""
  # Convert to disparity.
  disp = 1.0 / (depth + 1e-6)
  if normalizer is not None:
    disp /= normalizer
  else:
    disp /= (np.percentile(disp, pc) + 1e-6)
  disp = np.clip(disp, 0, 1)
  disp = _gray2rgb(disp, cmap=cmap)
  keep_h = int(disp.shape[0] * (1 - crop_percent))
  disp = disp[:keep_h]
  return disp

if __name__ == '__main__':
    disp_files = glob(args.predicted_disp_path + '/*.npy')
    disp_files.sort()
    output_dir=args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for disp in disp_files:

            pred_disparities = np.load(disp)
            num_samples = 1591

            pred_mask = []
            inputs = np.zeros((num_samples, args.img_height, args.img_width, 3),dtype=np.uint8)
            for t_id in range(num_samples):

                mask_pred = cv2.resize(pred_disparities[t_id][0], (args.img_width, args.img_height))
                mask_pred =  np.where(np.sqrt(mask_pred * mask_pred.shape[1])>=0.5, 0, 1)

                # need to convert from disparity to depth

                vertical_stack = 255*mask_pred
                mask_path = os.path.join(output_dir, '%03d.png' % t_id)

                scipy.misc.imsave(mask_path, vertical_stack)
                pred_mask.append(mask_pred)