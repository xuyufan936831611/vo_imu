#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from __future__ import division
import os
import numpy as np
import argparse
from glob import glob
from pose_evaluation_utils import *
import sys
parser = argparse.ArgumentParser()
parser.add_argument("--gtruth_dir", type=str,
    help='Path to the directory with ground-truth trajectories')
parser.add_argument("--pred_dir", type=str,
    help="Path to the directory with predicted trajectories")
args = parser.parse_args()

def main():
    path = os.listdir(args.pred_dir)
    seq=['09','10']
    for SEQ in seq:
        print('evaluate'+' '+SEQ+':')
        for p in path:
            up_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            pred_files = glob(up_path +'/'+args.pred_dir+ p +'/'+ SEQ + '/*.txt')
            pred_files.sort()
            pred_files.sort()
            ate_all = []

            for i in range(len(pred_files)):
                gtruth_file = args.gtruth_dir +SEQ+'/'+ os.path.basename(pred_files[i])
                if not os.path.exists(gtruth_file):
                    continue
                ate = compute_ate(gtruth_file, pred_files[i])
                if ate == False:
                    continue
                ate_all.append(ate)
            ate_all = np.array(ate_all)
            print('result of :%s'%p)
            print("ATE mean: %.4f, std: %.4f" % (np.mean(ate_all), np.std(ate_all)))
main()