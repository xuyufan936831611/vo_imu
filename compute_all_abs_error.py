# -*- coding:utf-8 -*-
import os
from glob import glob
import numpy as np
import sys
def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    x, y, z, w = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0/(Nq**2)
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ,                  xZ + wY],
         [xY + wZ,         1.0 - (xX + zZ),          yZ + wX],
         [xZ - wY,         yZ + wX,         1.0 - (xX + yY)]])

def main():
    pred_dir='pose_data_with_imu'
    write_rel_tra_dir='pose_rel'
    write_abs_tra_dir = 'pose_abs'
    path = os.listdir(pred_dir)  #pred_dir下的一级目录
    seq = ['09','10']
    for SEQ in seq:
        for p in path:
            print('create ' + SEQ+' abs path' + ' ' + 'of' + ' ' + p)
            pose_dir = sys.path[0] + '/' + pred_dir+'/' + p + '/' + SEQ
            N_sequence = len(glob(pose_dir + '/*.txt'))  # glob(img_dir + '/*.png'):查找在img_dir文件夹下的.png文件
            fns = [os.path.join(root, fn) for root, dirs, files in os.walk(pose_dir) for fn in files]
            fns.sort()
            if not os.path.isdir(sys.path[0] + '/' + write_rel_tra_dir + '/' + SEQ ):
                os.makedirs(sys.path[0] + '/' + write_rel_tra_dir + '/' + SEQ )
            w_img_dir = sys.path[0] + '/' + write_rel_tra_dir + '/' + SEQ + '/' + p + '.txt'
            with open(w_img_dir, 'w') as wf:
                for i in range(N_sequence):
                    if i == 0:
                        rf = open(fns[i], 'r')
                        content = rf.readlines()
                        wf.write(content[0])
                        wf.write(content[1])
                    # elif i==(N-1):
                    #    rf = open(fns[i], 'r')
                    #    content = rf.readlines()
                    #    thirs_lines = content[2]
                    #    wf.write(thirs_lines)
                    else:
                        rf = open(fns[i], 'r')
                        content = rf.readlines()
                        wf.write(content[1])
            tra_dir = w_img_dir
            true_dir = '/media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/备份/视觉里程计/dataset/KITTI/sequences/gt/dataset/poses/'+SEQ+'.txt'
            if not os.path.isdir(write_abs_tra_dir+'/' + p):
                os.makedirs(write_abs_tra_dir+'/' + p)
            mine_abs_dir = write_abs_tra_dir+'/' + p + '/'+SEQ+'.txt'
            if not os.path.exists(mine_abs_dir):
                os.mknod(mine_abs_dir)
            file = open(tra_dir)
            file_true = open(true_dir)
            with open(mine_abs_dir, 'w') as file_abs:
                contents = file.readlines()
                contents_true = file_true.readlines()
                N_sequence = len(contents)
                full_t = np.zeros((4, N_sequence), dtype=np.float32)
                pose = [[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]
                for i in range(N_sequence):
                    line = (contents[i]).split()
                    line_true_t = (contents_true[i]).split()
                    line_true_tp1 = (contents_true[i + 1]).split()
                    T_true_t = (np.reshape(np.array(line_true_t).astype(np.float), [3, 4]))
                    hfiller = np.array([0, 0, 0, 1]).reshape((1, 4))
                    T_true_t = np.concatenate((T_true_t, hfiller), axis=0)
                    T_true_tp1 = (np.reshape(np.array(line_true_tp1).astype(np.float), [3, 4]))
                    hfiller = np.array([0, 0, 0, 1]).reshape((1, 4))
                    T_true_tp1 = np.concatenate((T_true_tp1, hfiller), axis=0)
                    T_true_rel = -np.array(np.dot(np.linalg.inv(T_true_tp1), T_true_t)[0:3, 3])
                    T_result = (np.reshape(np.array(line[1:4]).astype(np.float), [3, 1]))

                    #############################################################
                    scale = 1
                    #############################################################
                    T_result = T_result * scale
                    q = np.array(line[4:8]).astype(np.float)
                    R = quat2mat(q)

                    Tmat = np.concatenate((R, T_result), axis=1)
                    hfiller = np.array([0, 0, 0, 1]).reshape((1, 4))
                    Tmat = np.concatenate((Tmat, hfiller), axis=0)

                    pose = np.dot(pose, Tmat)
                    file_abs.write(str(pose[0, 0]) + ' ' + str(pose[0, 1]) + ' ' + str(pose[0, 2]) + ' ' + str(pose[0, 3]) + ' ' + \
                                   str(pose[1, 0]) + ' ' + str(pose[1, 1]) + ' ' + str(pose[1, 2]) + ' ' + str(pose[1, 3]) + ' ' + \
                                   str(pose[2, 0]) + ' ' + str(pose[2, 1]) + ' ' + str(pose[2, 2]) + ' ' + str(pose[2, 3]))
                    file_abs.write('\n')
                    full_t[:, i] = pose[:, 3]

main()
