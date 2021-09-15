# Tensorflow 
## Tensorflow Installation (on Windows)
* Install **Python 3.8** for Windows
* Install nVidia **CUDA 11.1 & CuDNN 8.x** 
* Install **Tensorflow** 
  - download **[Tensorflow 2.4.0 cpu for Windows10](https://github.com/fo40225/tensorflow-windows-wheel/tree/master/2.4.0/py38/CPU%2BGPU/cuda111cudnn8avx2)** <br />
  - *`pip install tensorflow-2.4.0-cp38-cp38-win_amd64.whl`* <br />
* Download Examples *`git clone https://github.com/rkuo2000/tf`*
* Go to directory *`cd tf`*
## Tensorflow Sample Code
### Intro - Tensorflow Session & Tensorboard 
* *`jupyter notebook`* to run intro.ipynb
* run tensorboard : *`tensorboard --logdir=./`*
* use Chrome to open http://localhost:6006
### Basic Machine Learning
* one neuron network : *`python easy_net.py`*
* ten neurons network: *`python hidden_net.py`*
### MNIST : Handwritten Number Recognition
* plot data : *`python mnist_plotdata.py`*
* DNN : *`python mnist.py`*
* CNN : *`python mnist_cnn.py`*
* load model to predict : <br />
  *`python mnist_cnn_test.py`* (test data) <br />
  *`python mnist_cnn_image.py`* (image file) <br />
  *`python mnist_cnn_webcam.py`* (camera) <br />
### Fashion-MNIST : Fashion Wearing Recongition
* CNN : *`python fashionmnist_cnn.py`*
### Emotion Detection : Facial Expression Recognition
* Download the FER-2013 dataset from [here](https://anonfile.com/bdj3tfoeba/data_zip) and unzip it under data folder. 
* change directory name from data/data to data/fer2013
* To train the model, run *`python emotion_detection.py --mode train`*
* To detect facial expression, run *`python emotion_detection.py --mode detect`* 

### Edge TPU (USB Accelerator on RPi3B)
* **Convert _quant.tflite to _quant_edgetpu.tflite** <br />
* upload tl_mobilenetv2_quant.tflite to [EdgeTPU online compiler](https://coral.withgoogle.com/web-compiler/)<br />
* download _quant_edgetput.tflite and copy to RPi3 <br />
* **On RPi3B** <br />
*`cd ~`* <br />
*`git clone https://github.com/rkuo2000/tf`* (clone sample codes)<br />
*`cd ~/tf`* <br />
*`vi model/bird_labels.txt`* (create label file) <br />
* To test the model : <br />
*`python3 edgetpu_classify_webcam.py --model model/tl_mobilenetv2_quant.tflite --label model/bird_labels.txt`* <br />
*`python3 edgetpu_classify_image.py --model model/tl_mobilenetv2_quant.tflite --label model/bird_labels.txt`* <br />
