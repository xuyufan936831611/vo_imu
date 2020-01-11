#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import math
import numpy as np
import matplotlib.pyplot as plt

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
    test_seq='09'
    tra_dir='zhou_'+test_seq+'.txt'
    true_dir='/media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/备份/' \
             '视觉里程计/dataset/KITTI/sequences/gt/dataset/poses/'+test_seq+'.txt'
    mine_abs_dir='zhou_abs_'+test_seq+'.txt'
    file = open(tra_dir)
    file_true = open(true_dir)
    file_abs = open(mine_abs_dir,'w')
    contents = file.readlines()
    contents_true = file_true.readlines()
    N=len(contents)
    print(N)
    full_t = np.zeros((4, N), dtype=np.float32)
    pose=[[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1]]
    for i in range(N):
        line=(contents[i]).split()
        line_true_t = (contents_true[i]).split()
        line_true_tp1 = (contents_true[i+1]).split()
        T_true_t = (np.reshape(np.array(line_true_t).astype(np.float), [3, 4]))
        hfiller = np.array([0, 0, 0, 1]).reshape((1, 4))
        T_true_t = np.concatenate((T_true_t, hfiller), axis=0)
        T_true_tp1 = (np.reshape(np.array(line_true_tp1).astype(np.float), [3, 4]))
        hfiller = np.array([0, 0, 0, 1]).reshape((1, 4))
        T_true_tp1 = np.concatenate((T_true_tp1, hfiller), axis=0)
        T_true_rel = -np.array(np.dot(np.linalg.inv(T_true_tp1), T_true_t)[0:3, 3])
        T_result = -(np.reshape(np.array(line[1:4]).astype(np.float),[3,1]))
        if i!=0:
            scale = np.sum(T_true_rel * T_result) / np.sum(T_result ** 2)
        else:
            scale = 1
        #############################################################
        # scale = 1
        #############################################################
        T_result=T_result*scale
        q = np.array(line[4:8]).astype(np.float)
        R = quat2mat(q)

        Tmat = np.concatenate((R, T_result), axis=1)
        hfiller = np.array([0, 0, 0, 1]).reshape((1, 4))
        Tmat = np.concatenate((Tmat, hfiller), axis=0)

        pose = np.dot(pose,Tmat)
        file_abs.write(str(pose[0,0])+' '+str(pose[0,1])+' '+str(pose[0,2])+' '+str(pose[0,3])+' '+ \
                       str(pose[1,0])+' '+str(pose[1,1])+' '+str(pose[1,2])+' '+str(pose[1,3])+' '+ \
                       str(pose[2,0])+' '+str(pose[2,1])+' '+str(pose[2,2])+' '+str(pose[2,3]))
        file_abs.write('\n')
        full_t[ :,i] = pose[:,3]

    full_true_t = np.zeros((3, N), dtype=np.float32)
    for i in range(N):
        line=(contents_true[i]).split()
        T_true = (np.reshape(np.array([line[3],line[7],line[11]]).astype(np.float),3))
        full_true_t[ :,i] = T_true
    plt.plot(full_t[0,:], full_t[2,:],label=r'zhou ',lw=2)
    ############################################################################################################
    tra_dir = 'ours_'+test_seq+'_imu.txt'
    true_dir = '/media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/备份/视觉里程计/' \
               'dataset/KITTI/sequences/gt/dataset/poses/'+test_seq+'.txt'#model-239559
    mine_abs_dir = 'ours_abs_'+test_seq+'.txt'
    file = open(tra_dir)
    file_true = open(true_dir)
    file_abs = open(mine_abs_dir, 'w')
    contents = file.readlines()
    contents_true = file_true.readlines()
    N = len(contents)
    print(N)
    full_t = np.zeros((4, N), dtype=np.float32)
    pose = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
    for i in range(N):
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

    full_true_t = np.zeros((3, N), dtype=np.float32)
    for i in range(N):
        line = (contents_true[i]).split()
        T_true = (np.reshape(np.array([line[3], line[7], line[11]]).astype(np.float), 3))
        full_true_t[:, i] = T_true
    plt.plot(full_t[0, :], full_t[2, :], label=r'mine ', lw=2)
    plt.plot(full_true_t[0, :], full_true_t[2, :], label=r'true ', lw=2)
    plt.legend(loc='best', frameon=False)
    plt.show()
    plt.close()
main()
