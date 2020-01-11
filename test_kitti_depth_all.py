#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from __future__ import division
import tensorflow as tf
import numpy as np
import os
# import scipy.misc
import PIL.Image as pil
from SfMLearner import SfMLearner
from glob import glob

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string("ckpt_dir", None, "checkpoint file dir")
FLAGS = flags.FLAGS

def main(_):
    with open('data/kitti/test_files_eigen.txt', 'r') as f:
        test_files = f.readlines()
        test_files = [FLAGS.dataset_dir + t[:-1] for t in test_files]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    sfm = SfMLearner()
    sfm.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        batch_size=FLAGS.batch_size,
                        mode='depth')
    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    checkpoint_files = glob(FLAGS.ckpt_dir + '/*.meta')
    checkpoint_files.sort()
    for ckpt_file in checkpoint_files:
        ckpt_file=ckpt_file[:-5]
        basename = os.path.basename(ckpt_file)
        basename = FLAGS.ckpt_dir +'/'+ basename
        start_idx = basename.index('/')
        output_file = FLAGS.output_dir  + basename[start_idx+1:]
        with tf.Session(config=config) as sess:
            saver.restore(sess, basename)
            pred_all = []
            for t in range(0, len(test_files), FLAGS.batch_size):
                if t % 100 == 0:
                    print('processing %s: %d/%d' % (basename, t, len(test_files)))
                inputs = np.zeros(
                    (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3),
                    dtype=np.uint8)
                for b in range(FLAGS.batch_size):
                    idx = t + b
                    if idx >= len(test_files):
                        break
                    fh = open(test_files[idx], 'r')
                    raw_im = pil.open(fh)
                    scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                    inputs[b] = np.array(scaled_im)
                    # im = scipy.misc.imread(test_files[idx])
                    # inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
                pred = sfm.inference(inputs, sess, mode='depth')
                for b in range(FLAGS.batch_size):
                    idx = t + b
                    if idx >= len(test_files):
                        break
                    pred_all.append(pred['depth'][b,:,:])
            np.save(output_file, pred_all)

if __name__ == '__main__':
    tf.app.run()