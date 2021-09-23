import torch

print(torch.randn(10).to("cuda:0"))

import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))