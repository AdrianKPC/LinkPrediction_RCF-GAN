#import tensorflow as tf
#
#devices = tf.config.list_physical_devices()
#print(devices)
#tf.debugging.set_log_device_placement(True)
#a=tf.random.normal([100,100])
#b=tf.random.normal([100,100])
#c = a*b

import torch
import math
# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

#if not torch.backends.mps.is_available():
#    if not torch.backends.mps.is_built():
#        print("MPS not available because the current PyTorch install was not "
#              "built with MPS enabled.")
#    else:
#        print("MPS not available because the current MacOS version is not 12.3+ "
#              "and/or you do not have an MPS-enabled device on this machine.")
