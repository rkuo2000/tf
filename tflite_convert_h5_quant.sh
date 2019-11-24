### convert keras model .h5 to quantized .tflite (for edge TPU)
# cd ~/tf
# ./tflite_convert_h5_quant.sh tl_mobilenetv2
tflite_convert \
--keras_model_file=model/$1.h5 \
--output_file=model/$1_quant.tflite \
--inference_type=QUANTIZED_UINT8 \
--default_ranges_min=0 \
--default_ranges_max=255 \
--mean_values=128 \
--std_dev_values=128 \
--allow_custom_ops
