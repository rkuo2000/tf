""" for RPi3/4 + EdgeTPU   
Usage: 
cd ~/tf
python3 edgetpu_classify_webcam.py \
    --model model/inception_v4_299_quant_edgetpu.tflite  \
    --label model/imagenet_labels.txt

"""
import argparse
import io
import time
import numpy as np
import cv2

import edgetpu.classification.engine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
      '--label', help='File path of label file.', required=True)
    args = parser.parse_args()

    with open(args.label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    engine = edgetpu.classification.engine.ClassificationEngine(args.model)

    try:
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        _, width, height, channels = engine.get_input_tensor_shape()
        while True:
            ret, frame = cap.read()

            # Display the resulting frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            input = np.frombuffer(resized, dtype=np.uint8)
            start_time = time.time()
            results = engine.ClassifyWithInputTensor(input, top_k=1)
            elapsed_time = time.time() - start_time
            if results:
                confidence = results[0][1]
                label = labels[results[0][0]]
                print("Elapsed time: {:0.02f}".format(elapsed_time * 1000))
            cv2.putText(frame, label, (0, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "{:0.02f}".format(confidence), (0, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
