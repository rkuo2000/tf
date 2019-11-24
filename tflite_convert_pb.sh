### from tflite_graph.pb to detect.tflite
#cd ~/models/research/object_detection
#./tflite_convert_pb.sh training
tflite_convert \
--graph_def_file=$1/tflite_graph.pb \
--output_file=$1/converted_model.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=FLOAT \
--allow_custom_ops
