tflite_convert \
--keras_model_file=model/$1.h5 \
--output_file=model/$1_quant.tflite \
--inference_type=QUANTIZED_UINT8 \
--default_ranges_min=0 \
--default_ranges_max=255 \
--mean_values=128 \
--std_dev_values=128 \
--allow_custom_ops