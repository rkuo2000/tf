### from model.ckpt to tflite_graph.pb & .pbtxt
#cd ~/models/research/object_detection
#./tflite_export.sh training
python export_tflite_ssd_graph.py --pipeline_config_path $1/pipeline.config --trained_checkpoint_prefix $1/model.ckpt --output_directory $1 
