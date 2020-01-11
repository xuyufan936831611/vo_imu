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
    output_dir_gt = args.output_dir_gt
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # if not os.path.exists(output_dir_gt):
    #     os.makedirs(output_dir_gt)
    for disp in disp_files:

        if disp=="/media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/svo/depth_with_flow/model-181152.npy" or \
                disp=='depth_with_flow/model-181152.npy':

            pred_disparities = np.load(disp)
            if args.split == 'kitti':
                num_samples = 200

                gt_disparities = load_gt_disp_kitti(args.gt_path)
                gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities,
                                                                                                 pred_disparities)

            elif args.split == 'eigen':
                num_samples = 697
                test_files = read_text_lines(args.gt_path + 'new_eigen_test_files.txt')
                gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args.gt_path)

                num_test = len(im_files)
                gt_depths = []
                pred_depths = []
                inputs = np.zeros((num_samples, args.img_height, args.img_width, 3),dtype=np.uint8)
                for t_id in range(num_samples):
                    camera_id = cams[t_id]
                    depth,depth_interp = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, True, True)
                    gt_depths.append(depth.astype(np.float32))
                    disp_pred = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]),
                                           interpolation=cv2.INTER_LINEAR)
                    disp_pred = disp_pred * disp_pred.shape[1]
                    # need to convert from disparity to depth
                    focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
                    depth_pred = (baseline * focal_length) / disp_pred
                    depth_pred[np.isinf(depth_pred)] = 0
                    depth_map = np.squeeze(depth_pred)
                    colored_map = _normalize_depth_for_display(depth_map, cmap=CMAP)
                    input_float = inputs[t_id].astype(np.float32) / 255.0
                    vertical_stack = cv2.resize(colored_map,(1242,375))
                    depth_path = os.path.join(output_dir, '%03d.png' % t_id)
                    scipy.misc.imsave(depth_path, vertical_stack)
                    pred_depths.append(depth_pred)
                    ##########################################################
                    depth_interp[np.isinf(depth_interp)] = 0
                    white = 255 * np.ones([int(0.40 * args.img_height), args.img_width])
                    depth_display_gt = depth_interp.astype(np.float32)[int(0.40 * args.img_height):, :]
                    # depth_display_pred = 1. / (depth_display_gt+0.01)
                    depth_display_gt = cv2.normalize(depth_display_gt, depth_display_gt, alpha=0, beta=255,
                                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                    depth_gt = cv2.resize(depth_display_gt,(1242,int(0.60 *375)))
                    colored_gt_map = _normalize_depth_for_display(depth_gt, cmap=CMAP)
                    # depth_path_gt = os.path.join(output_dir_gt, '%03d.png' % t_id)
                    # scipy.misc.imsave(depth_path_gt, colored_gt_map)
                    ##########################################################

            rms = np.zeros(num_samples, np.float32)
            log_rms = np.zeros(num_samples, np.float32)
            abs_rel = np.zeros(num_samples, np.float32)
            sq_rel = np.zeros(num_samples, np.float32)
            d1_all = np.zeros(num_samples, np.float32)
            a1 = np.zeros(num_samples, np.float32)
            a2 = np.zeros(num_samples, np.float32)
            a3 = np.zeros(num_samples, np.float32)

            for i in range(num_samples):

                gt_depth = gt_depths[i]
                pred_depth = pred_depths[i]

                pred_depth[pred_depth < args.min_depth] = args.min_depth
                pred_depth[pred_depth > args.max_depth] = args.max_depth

                if args.split == 'eigen':
                    mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)

                    if args.garg_crop or args.eigen_crop:
                        gt_height, gt_width = gt_depth.shape

                        # crop used by Garg ECCV16
                        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                        if args.garg_crop:
                            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

                        # crop we found by trial and error to reproduce Eigen NIPS14 results
                        elif args.eigen_crop:
                            crop = np.array([0.3324324 * gt_height, 0.91351351 * gt_height,
                                             0.0359477 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

                        crop_mask = np.zeros(mask.shape)
                        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                        mask = np.logical_and(mask, crop_mask)

                if args.split == 'kitti':
                    gt_disp = gt_disparities[i]
                    mask = gt_disp > 0
                    pred_disp = pred_disparities_resized[i]

                    disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
                    bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
                    d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

                abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask],
                                                                                                pred_depth[mask])
            print('evaluation results of %s:'%disp)
            print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms',
                                                                                          'd1_all', 'a1', 'a2', 'a3'))
            print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(),
                                                                                                          sq_rel.mean(),
                                                                                                          rms.mean(),
                                                                                                          log_rms.mean(),
                                                                                                          d1_all.mean(),
                                                                                                          a1.mean(), a2.mean(),
                                                                                                          a3.mean()))
