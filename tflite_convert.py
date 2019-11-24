### TFLite Convert to .tflite
import pathlib
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.models import load_model

## for GPU
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

# Load Model 
#export_dir = 'model/saved_model'
model=load_model('model/tl_worms4.h5')

# TFLite Converter 
#converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# write TFLite model
tflite_models_dir = pathlib.Path("model/")
tflite_model_file = tflite_models_dir/"tf_worms4.tfilte"
tflite_model_file.write_bytes(tflite_model)