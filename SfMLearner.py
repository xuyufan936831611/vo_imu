# -*- coding:utf-8 -*-
from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from nets import *
from utils import *
from evaluation_utils import *


class SfMLearner(object):
    def __init__(self):
        pass
    
    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir_l,
                            opt.dataset_dir_r,
                            opt.test_dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
        self.num_scales = opt.num_scales
        self.batch_size = opt.batch_size
        self.img_width = opt.img_width
        self.img_height = opt.img_height
        with tf.name_scope("data_loading"):
            tgt_image, src_image_stack, Itr, BF, intrinsics,mask_t_l,imu_rpy = loader.load_train_batch()
            tgt_image = self.preprocess_image(tgt_image)
            src_image_stack = self.preprocess_image(src_image_stack)
            Itr = self.preprocess_image(Itr)
            mask_t_l = tf.image.convert_image_dtype(mask_t_l, dtype=tf.float32)
        with tf.name_scope("depth_prediction"):
            pred_disp_l, pred_disp_r, end_points = disp_net(tgt_image, is_training=True, flag=None)
            # pred_disp_l_ts1, _, _ = disp_net(I_l_ts1, is_training=True, flag=True)
            # pred_disp_l_tp1, _, _ = disp_net(I_l_tp1, is_training=True, flag=True)

        with tf.name_scope("pose_prediction"):
            pred_poses, pred_mask, end_points = pose_net(tgt_image, src_image_stack,
                                                         do_mask=(opt.explain_reg_weight > 0), is_training=True,
                                                         flag=None)

        with tf.name_scope("compute_loss"):
            exp_loss = 0
            pixel_loss = 0
            smooth_loss = 0
            appearance_loss = 0
            consistency_loss = 0
            flow_consistency_loss = 0
            rot_angle_error_loss = 0
            pred_depth = []
            tgt_image_all = []
            right_image_all = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            ref_exp_mask_all=[]
            pred_mask_all=[]
            exp_mask_stack_all=[]
            disp_mask_all=[]
            for s in range(opt.num_scales):
                # Scale the source and target images for computing loss at the
                # according scale.
                bf = tf.squeeze((tf.transpose(BF)), axis=0)
                bf = tf.tile(bf, [int(self.img_height / (2 ** s)) * int(self.img_width / (2 ** s))])
                bf = tf.transpose(tf.reshape(bf, [int(self.img_height / (2 ** s) * self.img_width / (2 ** s)), self.batch_size]))
                bf = tf.reshape(bf,[self.batch_size, int(self.img_height / (2 ** s)), int(self.img_width / (2 ** s)), 1])

                pred_disp_l[s] = tf.reshape(pred_disp_l[s],
                                    [self.batch_size, int(self.img_height / (2 ** s)),int(self.img_width / (2 ** s)), 1])
                pred_disp_r[s] = tf.reshape(pred_disp_r[s],
                                    [self.batch_size, int(self.img_height / (2 ** s)),int(self.img_width / (2 ** s)), 1])
                # pred_disp_l_ts1[s] = tf.reshape(pred_disp_l_ts1[s],
                #                     [self.batch_size, int(self.img_height / (2 ** s)),int(self.img_width / (2 ** s)), 1])
                # pred_disp_l_tp1[s] = tf.reshape(pred_disp_l_tp1[s],
                #                     [self.batch_size, int(self.img_height / (2 ** s)),int(self.img_width / (2 ** s)), 1])
                current_pred_depth_l = bf / (pred_disp_l[s] * (self.img_width))
                # current_pred_depth_l_ts1 = bf / (pred_disp_l_ts1[s] * (self.img_width))
                # current_pred_depth_l_tp1 = bf / (pred_disp_l_tp1[s] * (self.img_width))

                curr_tgt_image = tf.image.resize_area(tgt_image,
                                                      [int(self.img_height / (2 ** s)), int(self.img_width / (2 ** s))])
                curr_right_image = tf.image.resize_area(Itr,
                                                        [int(self.img_height / (2 ** s)),
                                                         int(self.img_width / (2 ** s))])
                curr_src_image_stack = tf.image.resize_area(src_image_stack,
                                                            [int(self.img_height / (2 ** s)),
                                                             int(self.img_width / (2 ** s))])
                ###########################################################################################
                I_t_l_,_ = self.generate_image_left(curr_right_image, pred_disp_l[s])
                I_t_r_,_ = self.generate_image_right(curr_tgt_image, pred_disp_r[s])
                mask_l = self.get_left_or_right_mask(s, do_reverse=False)
                mask_r = self.get_left_or_right_mask(s, do_reverse=True)
                consistency_loss = consistency_loss + self.compute_disp_loss(s,pred_disp_l[s], pred_disp_r[s])
                appearance_loss = appearance_loss + self.compute_appearance_loss(I_t_l_, I_t_r_,
                                                                                 curr_tgt_image,
                                                                                 curr_right_image,
                                                                                 mask_l,mask_r)

                if opt.smooth_weight > 0:
                    smooth_loss += opt.smooth_weight / (2 ** s) * tf.reduce_mean(
                                       tf.abs(self.get_disparity_smoothness(pred_disp_l[s], curr_tgt_image)) + \
                                       tf.abs(self.get_disparity_smoothness(pred_disp_r[s], curr_right_image)))
                if opt.explain_reg_weight > 0:
                    ref_exp_mask = self.get_reference_explain_mask_flow(s,mask_t_l)
                for i in range(opt.num_source):
                    # Inverse warp the source image to the target image frame

                    curr_proj_image,mask_target = projective_inverse_warp(curr_src_image_stack[:, :, :, 3 * i:3 * (i + 1)],
                                                              tf.squeeze(current_pred_depth_l, axis=3),
                                                              pred_poses[:, i, :], intrinsics[:, s, :, :])
                    curr_proj_error = self.compute_proj_loss(curr_proj_image , curr_tgt_image)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

                    if opt.explain_reg_weight > 0:

                        curr_exp_logits = tf.slice(pred_mask[s],
                                                   [0, 0, 0, i*2],
                                                   [-1, -1, -1, 2])
                        exp_loss += opt.explain_reg_weight * self.compute_exp_reg_loss(curr_exp_logits,ref_exp_mask)
                        curr_exp = tf.nn.softmax(curr_exp_logits)
                    # Photo-consistency loss weighted by explainability
                    if opt.explain_reg_weight > 0:
                        pixel_loss += tf.reduce_mean(curr_proj_error * tf.expand_dims(curr_exp[:,:,:,1], -1))
                        ref_exp_mask_all.append(tf.expand_dims(ref_exp_mask[:, :, :, 1], -1))
                        pred_mask_all.append(tf.expand_dims(curr_exp[:, :, :, 1], -1))
                    else:
                        pixel_loss += tf.reduce_mean(curr_proj_error)
                    # Prepare images for tensorboard summaries
                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,1], -1)
                    else:
                        proj_image_stack = tf.concat([proj_image_stack, curr_proj_image],3)
                        proj_error_stack = tf.concat([proj_error_stack, curr_proj_error],3)
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.concat([exp_mask_stack,curr_exp],3)
                    rot_angle_error_loss += tf.reduce_mean(tf.abs(imu_rpy[:, :, i * 3 + 2] + pred_poses[:, i, 4]))

                tgt_image_all.append(curr_tgt_image)
                right_image_all.append(curr_right_image)
                pred_depth.append(current_pred_depth_l)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                disp_mask_all.append(mask_l)
                if opt.explain_reg_weight > 0:
                    exp_mask_stack_all.append(exp_mask_stack)
            total_loss = pixel_loss + smooth_loss + consistency_loss + appearance_loss + exp_loss+rot_angle_error_loss #+ flow_consistency_loss



        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
            # self.grads_and_vars = optim.compute_gradients(total_loss, 
            #                                               var_list=train_vars)
            # self.train_op = optim.apply_gradients(self.grads_and_vars)
            self.train_op = slim.learning.create_train_op(total_loss, optim)
            self.global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step, 
                                              self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.mask = tf.expand_dims(curr_exp[:, :, :, 1], -1)
        self.pred_depth = pred_depth
        self.disp = pred_disp_l
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.exp_loss = exp_loss
        self.total_loss = total_loss
        self.pixel_loss = pixel_loss
        self.smooth_loss = smooth_loss
        self.consistency_loss = consistency_loss
        self.flow_consistency_loss = flow_consistency_loss
        self.appearance_loss = appearance_loss
        self.tgt_image_all = tgt_image_all
        self.right_image_all = right_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        self.ref_exp_mask_all = ref_exp_mask_all
        self.pred_mask_all = pred_mask_all
        self.disp_mask_all = disp_mask_all
    def build_test_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir_l,
                            opt.dataset_dir_r,
                            opt.test_dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
        self.num_scales = opt.num_scales
        self.batch_size = opt.batch_size
        self.img_width = opt.img_width
        self.img_height = opt.img_height
        with tf.name_scope("data_loading"):
            tgt_image = loader.load_test_batch()
            tgt_image = self.preprocess_image(tgt_image)
        with tf.name_scope("depth_prediction"):
            pred_test_depth_l, _, _ = disp_net(tgt_image, is_training=False, flag=True)
            self.pred_test_depth = pred_test_depth_l[0]
    def get_reference_explain_mask_flow(self, downscaling,mask_t_l):
        opt = self.opt
        tmp = np.array([1])
        tmp = np.tile(tmp,
                               (opt.batch_size,
                                int(opt.img_height/(2**downscaling)),
                                int(opt.img_width/(2**downscaling)),
                                1))
        tmp = tf.constant(tmp, dtype=tf.float32)
        mask_t_l=tf.image.resize_bilinear(mask_t_l, [int(opt.img_height/(2**downscaling)),
                                                     int(opt.img_width/(2**downscaling))])

        rel_tmp = tmp-tf.reduce_mean(mask_t_l,3,keepdims=True)
        ref_exp_mask = tf.concat([rel_tmp,tf.reduce_mean(mask_t_l,3,keepdims=True)],3)

        return ref_exp_mask
    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp,
                               (opt.batch_size,
                                int(opt.img_height/(2**downscaling)),
                                int(opt.img_width/(2**downscaling)),
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask
    def test_depth(self,sess,ckpt_file):
        dataset_dir='/home/omnisky/xuyufan/视觉里程计/dataset/KITTI/raw_data/kitti_raw_eigen/'
        output_dir='depth_mask/'
        gt_path='/home/omnisky/xuyufan/视觉里程计/dataset/KITTI/raw_data/kitti_raw_eigen/'
        min_depth=1e-3
        max_depth=80
        fetches={}
        with open('data/kitti/test_files_eigen.txt', 'r') as f:
            test_files = f.readlines()
            test_files = [dataset_dir + t[:-1] for t in test_files]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        basename = os.path.basename(ckpt_file)
        output_file = output_dir + '/' + basename
        pred_all = []
        for t in range(0, self.batch_size, self.batch_size):
            fetches['depth'] = self.pred_test_depth
            results = sess.run(fetches)
            for b in range(self.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break

                pred_all.append(results['depth'][b, :, :])
        np.save(output_file, pred_all)
        print('depth test done!')
        pred_disparities = pred_all
        num_samples = 4
        test_files = read_text_lines(gt_path + 'four_eigen_test_files.txt')
        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, gt_path)

        num_test = len(im_files)
        gt_depths = []
        pred_depths = []
        for t_id in range(num_samples):
            camera_id = cams[t_id]
            depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
            gt_depths.append(depth.astype(np.float32))

            disp_pred = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]),interpolation=cv2.INTER_LINEAR)
            disp_pred = disp_pred * disp_pred.shape[1]

            # need to convert from disparity to depth
            focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
            depth_pred = (baseline * focal_length) / disp_pred
            depth_pred[np.isinf(depth_pred)] = 0

            pred_depths.append(depth_pred)

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

            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth


            mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)


            gt_height, gt_width = gt_depth.shape

            # crop used by Garg ECCV16
            # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask],
                                                                                            pred_depth[mask])
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms',
                                                                                      'log_rms', 'd1_all', 'a1', 'a2',
                                                                                      'a3'))
        print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(),
                                                                                                      sq_rel.mean(),
                                                                                                      rms.mean(),
                                                                                                      log_rms.mean(),
                                                                                                      d1_all.mean(),
                                                                                                      a1.mean(),
                                                                                                      a2.mean(),
                                                                                                      a3.mean()))
        self.depth_eval=[abs_rel.mean(),sq_rel.mean(),rms.mean(),log_rms.mean(),d1_all.mean(),a1.mean(),a2.mean(),a3.mean()]
    def flow_warp(self,src_img, flow):
        """ inverse warp a source image to the target image plane based on flow field
        Args:
          src_img: the source  image [batch, height_s, width_s, 3]
          flow: target image to source image flow [batch, height_t, width_t, 2]
        Returns:
          Source image inverse warped to the target image plane [batch, height_t, width_t, 3]
        """
        batch, height, width, _ = src_img.get_shape().as_list()
        tgt_pixel_coords = tf.transpose(meshgrid(batch, height, width, False),
                                        [0, 2, 3, 1])
        src_pixel_coords = tgt_pixel_coords + flow
        output_img = bilinear_sampler(src_img, src_pixel_coords)
        return output_img
    def compute_rigid_flow(self,depth, pose, intrinsics, reverse_pose=False):
        """Compute the rigid flow from target image plane to source image

        Args:
          depth: depth map of the target image [batch, height_t, width_t]
          pose: target to source (or source to target if reverse_pose=True)
                camera transformation matrix [batch, 6], in the order of
                tx, ty, tz, rx, ry, rz;
          intrinsics: camera intrinsics [batch, 3, 3]
        Returns:
          Rigid flow from target image to source image [batch, height_t, width_t, 2]
        """
        batch, height, width = depth.get_shape().as_list()
        # Convert pose vector to matrix
        pose = pose_vec2mat(pose)
        if reverse_pose:
            pose = tf.matrix_inverse(pose)
        # Construct pixel grid coordinates
        pixel_coords = meshgrid(batch, height, width)
        tgt_pixel_coords = tf.transpose(pixel_coords[:, :2, :, :], [0, 2, 3, 1])
        # Convert pixel coordinates to the camera frame
        cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
        # Construct a 4x4 intrinsic matrix
        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch, 1, 1])
        intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])],2)
        intrinsics = tf.concat([intrinsics, filler],1)
        # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
        # pixel frame.
        proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
        src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
        rigid_flow = src_pixel_coords - tgt_pixel_coords
        return rigid_flow
    def compute_exp_reg_loss(self, pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)
    def compute_smooth_loss(self, pred_disp):
        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy
        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return tf.reduce_mean(tf.abs(dx2)) + \
               tf.reduce_mean(tf.abs(dxdy)) + \
               tf.reduce_mean(tf.abs(dydx)) + \
               tf.reduce_mean(tf.abs(dy2))
    def get_disparity_smoothness(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return smoothness_x + smoothness_y
    def gradient_x(self, img):
        batch,height,width,channel=img.get_shape().as_list()
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        edge=tf.zeros([batch,height,1,channel])
        gx=tf.concat([gx,edge],2)
        return gx
    def gradient_y(self, img):
        batch, height, width, channel = img.get_shape().as_list()
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        edge = tf.zeros([batch, 1, width, channel])
        gy = tf.concat([gy,edge ],1)
        return gy
    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("exp_loss", self.exp_loss)
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("consistency_loss", self.consistency_loss)
        tf.summary.scalar("appearance_loss", self.appearance_loss)
        tf.summary.scalar("flow_consistency_loss", self.flow_consistency_loss)
        for s in range(opt.num_scales):
            tf.summary.histogram("scale%d_depth" % s, self.pred_depth[s])
            tf.summary.image("scale%ddisp_mask_all" % s, self.disp_mask_all[s],max_outputs=16)
            #tf.summary.histogram("scale%d_disp" % s, self.disp[s])
            tf.summary.image('scale%d_disparity_image' % s,self.disp[s],max_outputs=16)
            tf.summary.image('scale%d_target_image' % s, \
                             self.deprocess_image(self.tgt_image_all[s]),max_outputs=16)
            if opt.explain_reg_weight > 0:
                # tf.summary.image(
                #     'scale%dpred_exp_mask_%d' % (s, i),
                #     tf.expand_dims(self.exp_mask_stack_all[s][:,:,:,i], -1))
                tf.summary.image(
                    'scale%dref_exp_mask' % (s), self.ref_exp_mask_all[s], max_outputs=16)
                tf.summary.image(
                    'scale%dpred_mask' % (s), self.pred_mask_all[s], max_outputs=16)
            # tf.summary.image('scale%d_right_image' % s, \
            #                  self.deprocess_image(self.right_image_all[s]),max_outputs=16)
            # for i in range(opt.num_source):
            #     tf.summary.image(
            #         'scale%d_source_image_%d' % (s, i),
            #         self.deprocess_image(self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3]),max_outputs=16)
            #     tf.summary.image('scale%d_projected_image_%d' % (s, i),
            #         self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3]),max_outputs=16)
            #     tf.summary.image('scale%d_proj_error_%d' % (s, i),
            #         self.deprocess_image(tf.clip_by_value(self.proj_error_stack_all[s][:,:,:,i*3:(i+1)*3] - 1, -1, 1)),max_outputs=4)
        tf.summary.histogram("tx", self.pred_poses[:,:,0])
        tf.summary.histogram("ty", self.pred_poses[:,:,1])
        tf.summary.histogram("tz", self.pred_poses[:,:,2])
        tf.summary.histogram("rx", self.pred_poses[:,:,3])
        tf.summary.histogram("ry", self.pred_poses[:,:,4])
        tf.summary.histogram("rz", self.pred_poses[:,:,5])
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        # for grad, var in self.grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)

    def train(self, opt):
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        opt.num_scales = 4
        self.opt = opt
        self.build_train_graph()
        #self.build_test_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=40)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                 save_summaries_secs=0, 
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(1, opt.max_steps):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step,

                }

                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)

                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/opt.summary_freq, 
                                results["loss"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)
                    # self.test_depth(sess, opt.checkpoint_dir)
    def build_mask_test_graph(self):

        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                                                self.img_height, self.img_width * 3, 3],
                                     name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, 2)
        with tf.name_scope("pose_prediction"):
            pred_poses, pred_mask, _ = pose_net(tgt_image, src_image_stack,do_mask=True, is_training=False)
            self.inputs = input_uint8
            self.pred_poses = pred_poses

        curr_exp_logits_f = tf.slice(pred_mask[0],
                                   [0, 0, 0, 0],
                                   [-1, -1, -1, 2])
        curr_exp_logits_b = tf.slice(pred_mask[0],
                                   [0, 0, 0, 2],
                                   [-1, -1, -1, 2])
        curr_exp_f = tf.nn.softmax(curr_exp_logits_f)
        curr_exp_b = tf.nn.softmax(curr_exp_logits_b)

        self.mask = tf.concat([tf.expand_dims(curr_exp_f[:,:,:,0],-1),tf.expand_dims(curr_exp_b[:,:,:,0],-1)],axis=3)

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
                    self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            pred_depth_l, pred_depth_r,_ = disp_net(
                input_mc, is_training=False)
        pred_depth = pred_depth_l[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth

    def build_pose_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
            self.img_height, self.img_width * self.seq_length, 3], 
            name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _ , _= pose_net(tgt_image, src_image_stack, is_training=False)
            self.inputs = input_uint8
            self.pred_poses = pred_poses

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self, 
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'mask':
            self.build_mask_test_graph()
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}

        if mode == 'mask':
            fetches['mask'] = self.mask
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
    #######################################################################
    def compute_disp_loss(self,s, D_l, D_r):
        opt = self.opt
        right_to_left_disp,_ = self.generate_image_left(D_r, D_l)
        left_to_right_disp,_ = self.generate_image_right(D_l, D_r)
        mask_disp_l = self.get_left_or_right_mask(s, do_reverse=False)
        mask_disp_r = self.get_left_or_right_mask(s, do_reverse=True)
        if opt.do_outliers == True:
            lr_left_loss = mask_disp_l * tf.abs(right_to_left_disp - D_l)
            lr_right_loss = mask_disp_r * tf.abs(left_to_right_disp - D_r)
        else:
            lr_left_loss = tf.abs(right_to_left_disp - D_l)
            lr_right_loss = tf.abs(left_to_right_disp - D_r)
        disp_lr_loss = lr_left_loss + lr_right_loss
        disp_lr_loss = tf.reduce_mean(disp_lr_loss)
        return disp_lr_loss
    #######################################################################

    def compute_appearance_loss(self, pred_l, pred_r, ref_l, ref_r,mask_l,mask_r):
        opt=self.opt
        lamada_s = 0.85
        ssim_loss_l = self.ssim(ref_l, pred_l)
        ssim_loss_l =tf.pad(ssim_loss_l,[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssim_loss_r = self.ssim(ref_r, pred_r)
        ssim_loss_r = tf.pad(ssim_loss_r, [[0, 0], [1, 1], [1, 1], [0, 0]])
        if opt.do_outliers==True:
            L_ssim_l = tf.reduce_mean(mask_l * tf.abs(ssim_loss_l))
            L_norm_l = tf.reduce_mean(mask_l * tf.abs(ref_l - pred_l))
            L_ssim_r = tf.reduce_mean(mask_r*tf.abs(ssim_loss_r))
            L_norm_r = tf.reduce_mean(mask_r*tf.abs(ref_r - pred_r))
        else:
            L_ssim_l = tf.reduce_mean(tf.abs(ssim_loss_l))
            L_norm_l = tf.reduce_mean(tf.abs(ref_l - pred_l))
            L_ssim_r = tf.reduce_mean(tf.abs(ssim_loss_r))
            L_norm_r = tf.reduce_mean(tf.abs(ref_r - pred_r))

        pho_consistency_lr_loss = lamada_s * (L_ssim_l + L_ssim_r) + (1 - lamada_s) * (L_norm_l + L_norm_r)

        return pho_consistency_lr_loss
    #######################################################################
    def compute_proj_loss(self, pred, ref):

        lamada_s = 0.85
        ssim_loss = self.ssim(ref, pred)
        ssim_loss =tf.pad(ssim_loss,[[0, 0], [1, 1], [1, 1], [0, 0]])

        L_ssim = tf.abs(ssim_loss)
        L_norm = tf.abs(ref - pred)

        pho_consistency_fb_loss = lamada_s * L_ssim  + (1 - lamada_s) * L_norm

        return pho_consistency_fb_loss
    def ssim(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')
        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y **2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / (SSIM_d+1e-8)

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_left_or_right_mask(self, downscaling, do_reverse=False):
        opt = self.opt
        boundery_size = 32
        tmp_0 = np.array([0])
        tmp_1 = np.array([1])
        mask_0 = np.tile(tmp_0,
                         (opt.batch_size,
                          int(opt.img_height / (2 ** downscaling)),
                          int(boundery_size / (2 ** downscaling)),
                          1))
        mask_1 = np.tile(tmp_1,
                         (opt.batch_size,
                          int(opt.img_height / (2 ** downscaling)),
                          int((opt.img_width - boundery_size) / (2 ** downscaling)),
                          1))
        if do_reverse == False:
            mask_all = np.concatenate((mask_0, mask_1),axis=2)
        else:
            mask_all = np.concatenate((mask_1, mask_0), axis=2)
        ref_exp_mask = tf.constant(mask_all, dtype=tf.float32)
        return ref_exp_mask
    def generate_image_left(self, img, disp):
        return self.bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return self.bilinear_sampler_1d_h(img, disp)

    def bilinear_sampler_1d_h(self,input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
        def _repeat(x, n_repeats):
            with tf.variable_scope('_repeat'):
                rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
                return tf.reshape(rep, [-1])

        def _interpolate(im, x, y):
            with tf.variable_scope('_interpolate'):

                # handle both texture border types
                _edge_size = 0
                if _wrap_mode == 'border':
                    _edge_size = 1
                    im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                    x = x + _edge_size
                    y = y + _edge_size
                elif _wrap_mode == 'edge':
                    _edge_size = 0
                else:
                    return None
                #######################################################################################
                X=tf.expand_dims(tf.cast(x,tf.float32),-1)
                X=tf.concat([tf.less(0.,X),tf.less(X, tf.cast(_width_f - 1 + 2 * _edge_size,tf.float32)),
                               tf.less(0.,X+1), tf.less(X+1, tf.cast(_width_f - 1 + 2 * _edge_size, tf.float32))],1)
                mask =tf.reduce_all(X,axis=1)
                #######################################################################################
                x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

                x0_f = tf.floor(x)
                y0_f = tf.floor(y)
                x1_f = x0_f + 1

                x0 = tf.cast(x0_f, tf.int32)
                y0 = tf.cast(y0_f, tf.int32)
                x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)

                dim2 = (_width + 2 * _edge_size)
                dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
                base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
                base_y0 = base + y0 * dim2
                idx_l = base_y0 + x0
                idx_r = base_y0 + x1

                im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

                pix_l = tf.gather(im_flat, idx_l)
                pix_r = tf.gather(im_flat, idx_r)

                weight_l = tf.expand_dims(x1_f - x, 1)
                weight_r = tf.expand_dims(x - x0_f, 1)
                ##########################################################################################
                return weight_l * pix_l + weight_r * pix_r,mask
                ##########################################################################################
        def _transform(input_images, x_offset):
            with tf.variable_scope('transform'):
                # grid of (x_t, y_t, 1), eq (1) in ref [1]
                x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                       tf.linspace(0.0 , _height_f - 1.0 , _height))

                x_t_flat = tf.reshape(x_t, (1, -1))
                y_t_flat = tf.reshape(y_t, (1, -1))

                x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
                y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

                x_t_flat = tf.reshape(x_t_flat, [-1])
                y_t_flat = tf.reshape(y_t_flat, [-1])

                x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f        #########################

                input_transformed ,mask= _interpolate(input_images, x_t_flat, y_t_flat)

                output = tf.reshape(
                    input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
                mask = tf.cast(tf.reshape(
                    mask, tf.stack([_num_batch, _height, _width, 1])),tf.float32)
                return output,mask

        with tf.variable_scope(name):
            _num_batch    = tf.shape(input_images)[0]
            _height       = tf.shape(input_images)[1]
            _width        = tf.shape(input_images)[2]
            _num_channels = tf.shape(input_images)[3]

            _height_f = tf.cast(_height, tf.float32)
            _width_f  = tf.cast(_width,  tf.float32)

            _wrap_mode = wrap_mode

            output,mask = _transform(input_images, x_offset)
            return output,mask
