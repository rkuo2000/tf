### https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
import numpy as np
import cv2
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to CheckPoint (frozen inference graph)
#MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
#MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
FROZEN_GRAPH_FILE = MODEL_NAME+'/frozen_inference_graph.pb'
LABEL_MAP_FILE = 'data/mscoco_label_map.pbtxt'
IMAGE_NAME = 'image1.jpg'
TEST_IMAGE_FILE = 'test_images/'+IMAGE_NAME

NUM_CLASSES = 90

# Load the label map (indices to category names)
label_map = label_map_util.load_labelmap(LABEL_MAP_FILE)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load Model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FROZEN_GRAPH_FILE, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph, config=config)

# Define input and output tensors (for image classifier)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# score is level of confidence
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load test image
image = cv2.imread(TEST_IMAGE_FILE)
image_expanded = np.expand_dims(image, axis=0)

# Perform Detection
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw Results of Detection
vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.60)

cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
