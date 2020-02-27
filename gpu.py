### Tensorflow 2.1
### https://www.tensorflow.org/guide/gpu 

## Ensure you have the latest TensorFlow gpu release installed.
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

## 1. number fo available GPUs
print("\n>>> 1. GPU - check available GPUs")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print("\n>>> 2. GPU - Logging device placement")
## 2. Logging device placement
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)

print("\n>>> 3. GPU - Manual device placement")
## 3. Manual device placement
tf.debugging.set_log_device_placement(True)

# Place tensors on the CPU
with tf.device('/CPU:0'):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

c = tf.matmul(a, b)
print(c)

print("\n>>> 4. GPU - Limiting GPU memory growth")
## 4. Limiting GPU memory growth
# single GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
