import numpy as np
data_dir='09.txt'
sum_tra_err=0
sum_rot_err=0
read_file = open(data_dir,'r')
content= read_file.readlines()
N=len(content)
for i in range(N):
    line = (content[i]).split()
    dir=np.array(line).astype(np.float)
    error_rot = np.abs(float(dir[1]))
    error_tra = float(dir[2])
    sum_rot_err += error_rot
    sum_tra_err += error_tra
avg_rot_err= (sum_rot_err) / N
avg_tra_err= (sum_tra_err *100) / N
print('tra_error:',avg_tra_err)
print('rot_error:',avg_rot_err)