#cd ~/models/research/object_detection
#./tflite_convert_pb_quant.sh training
tflite_convert \
--graph_def_file=$1/tflite_graph.pb \
--output_file=$1/converted_model_quant.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=QUANTIZED_UINT8 \
--default_ranges_min=0 \
--default_ranges_max=255 \
--mean_values=128 \
--std_dev_values=128 \
--allow_custom_ops
