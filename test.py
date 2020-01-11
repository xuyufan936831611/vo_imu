# # _*_ coding:utf-8 _*_
import torch
from torch.autograd import Variable as V
import os
from tensorflow.python import pywrap_tensorflow
model_dir=''
checkpoint_path = os.path.join(model_dir, "model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
  print("tensor_name: ", key)
  print(reader.get_tensor(key))
