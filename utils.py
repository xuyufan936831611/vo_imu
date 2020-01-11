from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='gray'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth

def euler2mat(z, y, x):
  """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """
  B = tf.shape(z)[0]
  N = 1
  z = tf.clip_by_value(z, -np.pi, np.pi)
  y = tf.clip_by_value(y, -np.pi, np.pi)
  x = tf.clip_by_value(x, -np.pi, np.pi)

  # Expand to B x N x 1 x 1
  z = tf.expand_dims(tf.expand_dims(z, -1), -1)
  y = tf.expand_dims(tf.expand_dims(y, -1), -1)
  x = tf.expand_dims(tf.expand_dims(x, -1), -1)

  zeros = tf.zeros([B, N, 1, 1])
  ones  = tf.ones([B, N, 1, 1])

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros],3)
  rotz_2 = tf.concat([sinz,  cosz, zeros],3)
  rotz_3 = tf.concat([zeros, zeros, ones],3)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3],2)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny],3)
  roty_2 = tf.concat([zeros, ones, zeros],3)
  roty_3 = tf.concat([-siny,zeros, cosy],3)
  ymat = tf.concat([roty_1, roty_2, roty_3],2)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros],3)
  rotx_2 = tf.concat([zeros, cosx, -sinx],3)
  rotx_3 = tf.concat([zeros, sinx, cosx],3)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3],2)

  rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
  return rotMat

def pose_vec2mat(vec):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
  batch_size, _ = vec.get_shape().as_list()
  translation = tf.slice(vec, [0, 0], [-1, 3])
  translation = tf.expand_dims(translation, -1)
  rx = tf.slice(vec, [0, 3], [-1, 1])
  ry = tf.slice(vec, [0, 4], [-1, 1])
  rz = tf.slice(vec, [0, 5], [-1, 1])
  rot_mat = euler2mat(rz, ry, rx)
  rot_mat = tf.squeeze(rot_mat, axis=[1])
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size, 1, 1])
  transform_mat = tf.concat([rot_mat, translation],2 )
  transform_mat = tf.concat([transform_mat, filler],1)
  return transform_mat
def lidaishu2mat(vec):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
  batch_size, _ = vec.get_shape().as_list()
  fai = tf.slice(vec, [0, 0], [-1, 3])
  fai = tf.expand_dims(fai, -1)#(4,3,1)
  theta = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(fai),axis=1)),axis=1)
  a = fai/theta
  rou=tf.expand_dims(tf.slice(vec, [0, 3], [-1, 3]),2)
  a_hat_0 = tf.expand_dims(tf.concat([tf.zeros([batch_size,1]),-a[:,2,:], a[:,1,:]],1),axis=1)
  a_hat_1 = tf.expand_dims(tf.concat([a[:,2,:],tf.zeros([batch_size,1]), -a[:,0,:]],1),axis=1)
  a_hat_2 = tf.expand_dims(tf.concat([-a[:,1,:],a[:,0,:] ,tf.zeros([batch_size, 1])],1),axis=1)
  a_hat =  tf.concat([a_hat_0, a_hat_1,a_hat_2],1)

  exp_fai_hat = tf.cos(theta) * tf.eye(3, batch_shape=[batch_size]) + \
                (tf.ones_like(theta) - tf.cos(theta)) * tf.matmul(a, tf.transpose(a,[0,2,1])) + \
                tf.sin(theta) * a_hat
  J = tf.sin(theta) / theta * tf.eye(3, batch_shape=[batch_size]) + \
                 (tf.ones_like(theta) - tf.sin(theta) / theta) * tf.matmul(a, tf.transpose(a,[0,2,1])) \
               + (tf.ones_like(theta) - tf.cos(theta)) / theta * a_hat
  t = tf.matmul(J, rou)
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size, 1, 1])
  transform_mat = tf.concat([exp_fai_hat, t],2 )
  transform_mat = tf.concat([transform_mat, filler],1)
  return transform_mat
def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  batch, height, width = depth.get_shape().as_list()
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
    cam_coords = tf.concat([cam_coords, ones],1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords

def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
  batch, _, height, width = cam_coords.get_shape().as_list()
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
  x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
  y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
  z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
  x_n = x_u / (z_u + 1e-10)
  y_n = y_u / (z_u + 1e-10)
  pixel_coords = tf.concat([x_n, y_n],1)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])

def meshgrid(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  if is_homogeneous:
    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis=0)
  else:
    coords = tf.stack([x_t, y_t], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  return coords

def projective_inverse_warp(img, depth, pose, intrinsics):
  """Inverse warp a source image to the target image plane based on projection.

  Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 6], in the
          order of tx, ty, tz, rx, ry, rz
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
  """
  batch, height, width, _ = img.get_shape().as_list()
  # Convert pose vector to matrix
  pose = pose_vec2mat(pose)
  # pose = lidaishu2mat(pose)
  # Construct pixel grid coordinates
  pixel_coords = meshgrid(batch, height, width)
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
  # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])],2)
  intrinsics = tf.concat([intrinsics, filler],1)
  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
  ######################################################################
  output_img,mask = bilinear_sampler(img, src_pixel_coords)
  return output_img,mask
  ######################################################################
def bilinear_sampler(imgs, coords):
  """Construct a new image by bilinear sampling from the input image.

  Points falling outside the source image boundary have value 0.

  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords,[1,1],3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    ############################################################################

    cords = tf.concat([tf.less(0., x0), tf.less(x0, x_max),
                          tf.less(0., y0), tf.less(y0, y_max),
                          tf.less(0., x1), tf.less(x1, x_max),
                          tf.less(0., y1), tf.less(y1, y_max)],3)

    mask = tf.reduce_all(cords, axis=3,keep_dims=True)
    mask= tf.cast(tf.reshape(mask,[out_size[0], out_size[1], out_size[2], 1]),tf.float32)
    ############################################################################
    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    return output,mask
def build_val_graph(self):
    opt = self.opt
    loader = DataLoader(opt.dataset_dir_l,
                        opt.dataset_dir_r,
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
        tgt_image, src_image_stack,Itr, BF,intrinsics = loader.load_val_batch()
        tgt_image = self.preprocess_image(tgt_image)
        src_image_stack = self.preprocess_image(src_image_stack)
        Itr = self.preprocess_image(Itr)
        stereo_img=tf.concat([tgt_image,Itr],3)
    with tf.name_scope("depth_prediction"):
        pred_disp_l,pred_disp_r,end_points = disp_net(tgt_image,is_training=True,flag=True)

    with tf.name_scope("pose_prediction"):
        pred_poses, pose_net_endpoints = pose_net(tgt_image,src_image_stack,is_training=True,flag=True)

    with tf.name_scope("compute_loss"):
        pixel_loss = 0
        smooth_loss = 0
        appearance_loss=0
        consistency_loss=0
        pred_depth=[]
        tgt_image_all = []
        right_image_all=[]
        src_image_stack_all = []
        proj_image_stack_all = []
        proj_error_stack_all = []
        for s in range(opt.num_scales):
            # Scale the source and target images for computing loss at the
            # according scale.
            bf = tf.squeeze((tf.transpose(BF)), axis=0)
            bf = tf.tile(bf, [int(self.img_height / (2 ** s)) * int(self.img_width / (2 ** s))])
            bf = tf.transpose(tf.reshape(bf, [int(self.img_height / (2 ** s) * self.img_width / (2 ** s)), self.batch_size]))
            bf = tf.reshape(bf,[self.batch_size, int(self.img_height / (2 ** s)), int(self.img_width / (2 ** s)), 1])

            pred_disp_l[s] = tf.reshape(pred_disp_l[s],
                                        [self.batch_size, int(self.img_height / (2 ** s)),
                                         int(self.img_width / (2 ** s)), 1])
            pred_disp_r[s] = tf.reshape(pred_disp_r[s],
                                        [self.batch_size, int(self.img_height / (2 ** s)),
                                         int(self.img_width / (2 ** s)), 1])
            current_pred_depth_l = bf / (pred_disp_l[s] * (self.img_width))
            current_pred_depth_r = bf / (pred_disp_r[s] * (self.img_width))
            curr_tgt_image = tf.image.resize_area(tgt_image,
                                                  [int(self.img_height / (2 ** s)), int(self.img_width / (2 ** s))])
            curr_right_image = tf.image.resize_area(Itr,
                                                    [int(self.img_height / (2 ** s)),
                                                     int(self.img_width / (2 ** s))])
            curr_src_image_stack = tf.image.resize_area(src_image_stack,
                                                        [int(self.img_height / (2 ** s)),
                                                         int(self.img_width / (2 ** s))])
            I_t_l_ = self.generate_image_left(curr_right_image, pred_disp_r[s])
            I_t_r_ = self.generate_image_right(curr_tgt_image, pred_disp_l[s])

            consistency_loss = consistency_loss + self.compute_disp_loss(pred_disp_l[s], pred_disp_r[s])
            appearance_loss = appearance_loss + self.compute_appearance_loss(I_t_l_, I_t_r_, curr_tgt_image, curr_right_image)
            if opt.smooth_weight > 0:
                smooth_loss += opt.smooth_weight / (2 ** s) * \
                           tf.reduce_mean(tf.abs(self.get_disparity_smoothness(pred_disp_l[s], curr_tgt_image))+\
                                          tf.abs(self.get_disparity_smoothness(pred_disp_r[s], curr_right_image)))
            for i in range(opt.num_source):
                # Inverse warp the source image to the target image frame
                curr_proj_image,mask_target = projective_inverse_warp(curr_src_image_stack[:,:,:,3*i:3*(i+1)],
                    tf.squeeze(current_pred_depth_l, axis=3),pred_poses[:,i,:],intrinsics[:,s,:,:])
                curr_proj_error = mask_target*tf.abs(curr_proj_image - curr_tgt_image)
                pixel_loss += tf.reduce_mean(curr_proj_error)
                # Prepare images for tensorboard summaries
                if i == 0:
                    proj_image_stack = curr_proj_image
                    proj_error_stack = curr_proj_error
                else:
                    proj_image_stack = tf.concat([proj_image_stack,curr_proj_image],3)
                    proj_error_stack = tf.concat([proj_error_stack,curr_proj_error],3)

            tgt_image_all.append(curr_tgt_image)
            right_image_all.append(curr_right_image)
            pred_depth.append(current_pred_depth_l)
            src_image_stack_all.append(curr_src_image_stack)
            proj_image_stack_all.append(proj_image_stack)
            proj_error_stack_all.append(proj_error_stack)
        total_val_loss = pixel_loss + smooth_loss + consistency_loss + appearance_loss
    self.total_val_loss=total_val_loss