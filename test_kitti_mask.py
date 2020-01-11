from __future__ import division
import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
from SfMLearner import SfMLearner
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
FLAGS = flags.FLAGS

def load_image_sequence(dataset_dir,
                        frames,
                        tgt_idx,
                        seq_length,
                        img_height,
                        img_width):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        curr_img = scipy.misc.imread(img_file)
        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        if o == -half_offset:
            image_seq = curr_img
        else:
            image_seq = np.hstack((image_seq, curr_img))
    return image_seq

def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    # TODO: unnecessary to check if the drives match
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False

def main():
    sfm = SfMLearner()
    sfm.setup_inference(FLAGS.img_height,
                        FLAGS.img_width,
                        'mask',
                        FLAGS.seq_length)
    saver = tf.train.Saver([var for var in tf.trainable_variables()])

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    seq_dir = os.path.join(FLAGS.dataset_dir, 'sequences', '%.2d' % FLAGS.test_seq)
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]
    with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
        times = f.readlines()
    pred_all=[]
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.ckpt_file)

        for tgt_idx in range(N):
            if not is_valid_sample(test_frames, tgt_idx, FLAGS.seq_length):
                continue
            if tgt_idx % 100 == 0:
                print('Progress: %d/%d' % (tgt_idx, N))
            # TODO: currently assuming batch_size = 1
            image_seq = load_image_sequence(FLAGS.dataset_dir,
                                            test_frames,
                                            tgt_idx,
                                            FLAGS.seq_length,
                                            FLAGS.img_height,
                                            FLAGS.img_width)
            pred = sfm.inference(image_seq[None, :, :, :], sess, mode='mask')
            pred_all.append(pred['mask'][:,:,:,0]*pred['mask'][:,:,:,1])
    basename = os.path.basename(FLAGS.ckpt_file)
    output_file = FLAGS.output_dir + '/' + basename
    np.save(output_file, pred_all)




main()