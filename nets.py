from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np

# Range of disparity/inverse depth values
DISP_SCALING = 0.3
MIN_DISP = 0.00001


def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


def pose_net(tgt_image, src_image_stack, do_mask=False, is_training=True,flag=None):
    inputs = tf.concat( [tgt_image,src_image_stack],3)
    b = tgt_image.get_shape()[0].value
    # print("batch_size:%d" % b)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value // 3)
    with tf.variable_scope('pose_net',reuse=flag) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1 = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride=2, scope='cnv2')
            cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3')
            cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose'):
                cnv6 = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                # cnv7a = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7a')
                # cnv7b = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7b')
                # pose_preda = slim.conv2d(cnv7a, 6 , [1, 1], scope='preda',
                #                         stride=1, normalizer_fn=None, activation_fn=None)
                # pose_predb = slim.conv2d(cnv7b, 6, [1, 1], scope='predb',
                #                         stride=1, normalizer_fn=None, activation_fn=None)
                # pose_avga = tf.reduce_mean(pose_preda, [1, 2])
                # pose_avgb = tf.reduce_mean(pose_predb, [1, 2])
                # # Empirically we found that scaling by a small constant
                # # facilitates training.
                # pose_finala = 0.01 * tf.reshape(pose_avga, [-1, 1, 6])
                # pose_finalb = 0.01 * tf.reshape(pose_avgb, [-1, 1, 6])
                # pose_final = tf.concat([pose_finala,pose_finalb],1)
                cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6 * num_source, [1, 1], scope='pred',
                                        stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                pose_final = 0.01 * tf.reshape(pose_avg, [-1,  num_source, 6 ])
            # Exp mask specific layers
            if do_mask:
                with tf.variable_scope('mask'):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                    mask4 = slim.conv2d(upcnv4, num_source * 2, [3, 3], stride=1, scope='mask4', normalizer_fn=None)
                    mask4_up = tf.image.resize_bilinear(mask4, [np.int(H / 4), np.int(W / 4)])

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64, [3, 3], stride=2, scope='upcnv3')

                    # i3_in = tf.concat( [upcnv3,mask4_up],3)
                    mask3 = slim.conv2d(upcnv3, num_source * 2, [3, 3], stride=1, scope='mask3', normalizer_fn=None)
                    mask3_up = tf.image.resize_bilinear(mask3, [np.int(H / 2), np.int(W / 2)])

                    upcnv2 = slim.conv2d_transpose(upcnv3, 32, [5, 5], stride=2, scope='upcnv2')
                    # i2_in = tf.concat( [upcnv2,mask3_up],3)
                    mask2 = slim.conv2d(upcnv2, num_source * 2, [5, 5], stride=1, scope='mask2', normalizer_fn=None)
                    mask2_up = tf.image.resize_bilinear(mask2, [np.int(H), np.int(W)])

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16, [7, 7], stride=2, scope='upcnv1')
                    # i1_in = tf.concat( [upcnv1,mask2_up],3)
                    mask1 = slim.conv2d(upcnv1, num_source * 2, [7, 7], stride=1, scope='mask1', normalizer_fn=None)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, [mask1, mask2, mask3, mask4], end_points


# def disp_net(tgt_image, is_training=True, flag=None):
#     H = tgt_image.get_shape()[1].value
#     W = tgt_image.get_shape()[2].value
#     with tf.variable_scope('disp_net', reuse=flag) as sc:
#         end_points_collection = sc.original_name_scope + '_end_points'
#         with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
#                             normalizer_fn=slim.batch_norm, # None,
#                             weights_regularizer=slim.l2_regularizer(0.05),
#                             activation_fn=tf.nn.relu,
#                             outputs_collections=end_points_collection):
#             cnv1 = slim.conv2d(tgt_image, 32, [7, 7], stride=2, scope='cnv1')
#             cnv1b = slim.conv2d(cnv1, 32, [7, 7], stride=1, scope='cnv1b')
#             # cnv1b = slim.avg_pool2d(cnv1b,2)
#             cnv2 = slim.conv2d(cnv1b, 64, [5, 5], stride=2, scope='cnv2')
#             cnv2b = slim.conv2d(cnv2, 64, [5, 5], stride=1, scope='cnv2b')
#             # cnv2b = slim.avg_pool2d(cnv2b, 2)
#             cnv3 = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
#             cnv3b = slim.conv2d(cnv3, 128, [3, 3], stride=1, scope='cnv3b')
#             # cnv3b = slim.avg_pool2d(cnv3b, 2)
#             cnv4 = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
#             cnv4b = slim.conv2d(cnv4, 256, [3, 3], stride=1, scope='cnv4b')
#             # cnv4b = slim.avg_pool2d(cnv4b, 2)
#             cnv5 = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
#             cnv5b = slim.conv2d(cnv5, 512, [3, 3], stride=1, scope='cnv5b')
#             # cnv5b = slim.avg_pool2d(cnv5b, 2)
#             cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
#             cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')
#             # cnv6b = slim.avg_pool2d(cnv6b, 2)
#             cnv7 = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
#             cnv7b = slim.conv2d(cnv7, 512, [3, 3], stride=1, scope='cnv7b')
#             # cnv7b = slim.avg_pool2d(cnv7b, 2)
#
#             upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
#             # upcnv7 = tf.image.resize_nearest_neighbor(cnv7b, [int(2*cnv7b.get_shape()[1].value), int(2*cnv7b.get_shape()[2].value)])
#             # upcnv7 = slim.conv2d(upcnv7, 512, [3, 3], stride=1, scope='upcnv7')
#             # There might be dimension mismatch due to uneven down/up-sampling
#             upcnv7 = resize_like(upcnv7, cnv6b)
#             i7_in = tf.concat([upcnv7, cnv6b], 3)
#             icnv7 = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')
#
#             upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
#             # upcnv6 = tf.image.resize_nearest_neighbor(icnv7, [int(2 * icnv7.get_shape()[1].value), int(2 * icnv7.get_shape()[2].value)])
#             # upcnv6 = slim.conv2d(upcnv6, 512, [3, 3], stride=1, scope='upcnv6')
#             upcnv6 = resize_like(upcnv6, cnv5b)
#             i6_in = tf.concat([upcnv6, cnv5b], 3)
#             icnv6 = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')
#
#             upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
#             # upcnv5 = tf.image.resize_nearest_neighbor(icnv6, [int(2 * icnv6.get_shape()[1].value), int(2 * icnv6.get_shape()[2].value)])
#             # upcnv5 = slim.conv2d(upcnv5, 256, [3, 3], stride=1, scope='upcnv5')
#             upcnv5 = resize_like(upcnv5, cnv4b)
#             i5_in = tf.concat( [upcnv5, cnv4b],3)
#             icnv5 = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')
#
#             upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
#             # upcnv4 = tf.image.resize_nearest_neighbor(icnv5,[int(2 * icnv5.get_shape()[1].value), int(2 * icnv5.get_shape()[2].value)])
#             # upcnv4 = slim.conv2d(upcnv4, 128, [3, 3], stride=1, scope='upcnv4')
#             i4_in = tf.concat( [upcnv4, cnv3b],3)
#             icnv4 = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
#             disp4 = DISP_SCALING * slim.conv2d(icnv4, 2, [3, 3], stride=1,
#                                                activation_fn=tf.sigmoid, normalizer_fn=None,
#                                                scope='disp4') + MIN_DISP  #
#             # disp4_up = tf.image.resize_bilinear(disp4, [np.int(H / 4), np.int(W / 4)])
#             disp4_up = tf.image.resize_bilinear(disp4,[np.int(H / 4), np.int(W / 4)])
#             disp4_up = slim.conv2d(disp4_up, 128, [3, 3], stride=1, scope='disp4_up')
#
#             upcnv3 = slim.conv2d_transpose(icnv4, 64, [3, 3], stride=2, scope='upcnv3')
#             # upcnv3 = tf.image.resize_nearest_neighbor(icnv4,[int(2 * icnv4.get_shape()[1].value), int(2 * icnv4.get_shape()[2].value)])
#             # upcnv3 = slim.conv2d(upcnv3, 64, [3, 3], stride=1, scope='upcnv3')
#             i3_in = tf.concat([upcnv3, cnv2b, disp4_up], 3)
#             icnv3 = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
#             disp3 = DISP_SCALING * slim.conv2d(icnv3, 2, [3, 3], stride=1,
#                                                activation_fn=tf.sigmoid, normalizer_fn=None,
#                                                scope='disp3') + MIN_DISP  #
#             # disp3_up = tf.image.resize_bilinear(disp3, [np.int(H / 2), np.int(W / 2)])
#             disp3_up = tf.image.resize_bilinear(disp3, [np.int(H / 2), np.int(W / 2)])
#             disp3_up = slim.conv2d(disp3_up, 128, [3, 3], stride=1, scope='disp3_up')
#
#             upcnv2 = slim.conv2d_transpose(icnv3, 32, [3, 3], stride=2, scope='upcnv2')
#             # upcnv2 = tf.image.resize_nearest_neighbor(icnv3,[int(2 * icnv3.get_shape()[1].value), int(2 * icnv3.get_shape()[2].value)])
#             # upcnv2 = slim.conv2d(upcnv2, 32, [3, 3], stride=1, scope='upcnv2')
#             i2_in = tf.concat([upcnv2, cnv1b, disp3_up], 3)
#             icnv2 = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
#             disp2 = DISP_SCALING * slim.conv2d(icnv2, 2, [3, 3], stride=1,
#                                                activation_fn=tf.sigmoid, normalizer_fn=None,
#                                                scope='disp2') + MIN_DISP  #
#             # disp2_up = tf.image.resize_bilinear(disp2, [H, W])
#             disp2_up = tf.image.resize_bilinear(disp2, [H, W])
#             disp2_up = slim.conv2d(disp2_up, 128, [3, 3], stride=1, scope='disp2_up')
#
#             upcnv1 = slim.conv2d_transpose(icnv2, 16, [3, 3], stride=2, scope='upcnv1')
#             # upcnv1 = tf.image.resize_nearest_neighbor(icnv2,[int(2 * icnv2.get_shape()[1].value), int(2 * icnv2.get_shape()[2].value)])
#             # upcnv1 = slim.conv2d(upcnv1, 16, [3, 3], stride=1, scope='upcnv1')
#             i1_in = tf.concat([upcnv1, disp2_up], 3)
#             icnv1 = slim.conv2d(i1_in, 16, [3, 3], stride=1, scope='icnv1')
#             disp1 = DISP_SCALING * slim.conv2d(icnv1, 2, [3, 3], stride=1,
#                                                activation_fn=tf.sigmoid, normalizer_fn=None,
#                                                scope='disp1') + MIN_DISP  #
#
#             end_points = utils.convert_collection_to_dict(end_points_collection)
#             return [disp1[:, :, :, 0], disp2[:, :, :, 0], disp3[:, :, :, 0], disp4[:, :, :, 0]], \
#                    [disp1[:, :, :, 1], disp2[:, :, :, 1], disp3[:, :, :, 1], disp4[:, :, :, 1]], end_points
def disp_net(tgt_image, is_training=True, flag=None):
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('disp_net', reuse=flag) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm, # None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1 = slim.conv2d(tgt_image, 32, [7, 7], stride=1, scope='cnv1')
            cnv1b = slim.conv2d(cnv1, 32, [7, 7], stride=1, scope='cnv1b')
            cnv1b = slim.avg_pool2d(cnv1b,2)
            cnv2 = slim.conv2d(cnv1b, 64, [5, 5], stride=1, scope='cnv2')
            cnv2b = slim.conv2d(cnv2, 64, [5, 5], stride=1, scope='cnv2b')
            cnv2b = slim.avg_pool2d(cnv2b, 2)
            cnv3 = slim.conv2d(cnv2b, 128, [3, 3], stride=1, scope='cnv3')
            cnv3b = slim.conv2d(cnv3, 128, [3, 3], stride=1, scope='cnv3b')
            cnv3b = slim.avg_pool2d(cnv3b, 2)
            cnv4 = slim.conv2d(cnv3b, 256, [3, 3], stride=1, scope='cnv4')
            cnv4b = slim.conv2d(cnv4, 256, [3, 3], stride=1, scope='cnv4b')
            cnv4b = slim.avg_pool2d(cnv4b, 2)
            cnv5 = slim.conv2d(cnv4b, 512, [3, 3], stride=1, scope='cnv5')
            cnv5b = slim.conv2d(cnv5, 512, [3, 3], stride=1, scope='cnv5b')
            cnv5b = slim.avg_pool2d(cnv5b, 2)
            cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=1, scope='cnv6')
            cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')
            cnv6b = slim.avg_pool2d(cnv6b, 2)
            cnv7 = slim.conv2d(cnv6b, 512, [3, 3], stride=1, scope='cnv7')
            cnv7b = slim.conv2d(cnv7, 512, [3, 3], stride=1, scope='cnv7b')
            cnv7b = slim.avg_pool2d(cnv7b, 2)

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in = tf.concat([upcnv7, cnv6b], 3)
            icnv7 = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in = tf.concat([upcnv6, cnv5b], 3)
            icnv6 = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in = tf.concat( [upcnv5, cnv4b],3)
            icnv5 = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in = tf.concat( [upcnv4, cnv3b],3)
            icnv4 = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4 = DISP_SCALING * slim.conv2d(icnv4, 2, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None,
                                               scope='disp4') + MIN_DISP  #
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H / 4), np.int(W / 4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 64, [3, 3], stride=2, scope='upcnv3')
            i3_in = tf.concat([upcnv3, cnv2b, disp4_up], 3)
            icnv3 = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
            disp3 = DISP_SCALING * slim.conv2d(icnv3, 2, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None,
                                               scope='disp3') + MIN_DISP  #
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H / 2), np.int(W / 2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32, [3, 3], stride=2, scope='upcnv2')
            i2_in = tf.concat([upcnv2, cnv1b, disp3_up], 3)
            icnv2 = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
            disp2 = DISP_SCALING * slim.conv2d(icnv2, 2, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None,
                                               scope='disp2') + MIN_DISP  #
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16, [3, 3], stride=2, scope='upcnv1')
            i1_in = tf.concat([upcnv1, disp2_up], 3)
            icnv1 = slim.conv2d(i1_in, 16, [3, 3], stride=1, scope='icnv1')
            disp1 = DISP_SCALING * slim.conv2d(icnv1, 2, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None,
                                               scope='disp1') + MIN_DISP  #

            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1[:, :, :, 0], disp2[:, :, :, 0], disp3[:, :, :, 0], disp4[:, :, :, 0]], \
                   [disp1[:, :, :, 1], disp2[:, :, :, 1], disp3[:, :, :, 1], disp4[:, :, :, 1]], end_points
