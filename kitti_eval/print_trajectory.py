import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
    tra_dir='their_10.txt'
    file = open(tra_dir)
    contents=file.readlines()
    N=len(contents)
    full_t = np.zeros((4, N), dtype=np.float32)
    pose=[[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1]]
    for i in range(N):
        line=(contents[i]).split()

        T = (np.reshape(np.array(line[1:4]).astype(np.float),[3,1]))
        q = np.array(line[4:8]).astype(np.float)
        R = quat2mat(q)

        Tmat = np.concatenate((R, T), axis=1)
        hfiller = np.array([0, 0, 0, 1]).reshape((1, 4))
        Tmat = np.concatenate((Tmat, hfiller), axis=0)

        pose = np.dot(pose,Tmat)
        full_t[ :,i] = pose[:,3]
        print(pose[:,3])
    plt.plot(full_t[0,:], full_t[2,:])
    plt.show()

main()
