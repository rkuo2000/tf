### convert keras model .h5 into .tflite (for PC/Android)
# cd ~/tf
# ./tflite_convert_h5.sh tl_mobilenetv2
tflite_convert \
--keras_model_file=model/$1.h5 \
--output_file=model/$1.tflite
