### Tensorflow 2.1
### https://www.tensorflow.org/guide/gpu 

## Ensure you have the latest TensorFlow gpu release installed.
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

## clear session & reset default graph
#tf.keras.backend.clear_session()
#tf.compat.v1.reset_default_graph()

## check number of available GPUs
print("\n>>> GPU : check available GPUs")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

## Limiting GPU memory growth
print("\n>>> GPU : Limiting GPU memory growth")
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

## Logging device placement 
print("\n>>> GPU : Logging device placement")
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)
