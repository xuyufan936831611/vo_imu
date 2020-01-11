# -*- coding:utf-8 -*-
import os
from glob import glob
r_img_dir='pose_data_with_imu/model-166532/10'#model-240002/09
N = len(glob(r_img_dir + '/*.txt'))  # glob(img_dir + '/*.png'):查找在img_dir文件夹下的.png文件
w_img_dir = 'ours_10_imu.txt'
fns = [os.path.join(root,fn) for root, dirs, files in os.walk(r_img_dir) for fn in files]
fns.sort()
wf=open(w_img_dir,'w')
for i in range(N):
  if i==0:
     rf = open(fns[i],'r')
     content=rf.readlines()
     first_lines=content[0]
     second_lines = content[1]
     wf.write(first_lines)
     wf.write(second_lines)
  # elif i==(N-1):
  #    rf = open(fns[i], 'r')
  #    content = rf.readlines()
  #    thirs_lines = content[2]
  #    wf.write(thirs_lines)
  else:
     rf = open(fns[i], 'r')
     content = rf.readlines()
     second_lines = content[1]
     wf.write(second_lines)

