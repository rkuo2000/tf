# Dataset
* Download dataset from website and copy zip file to ~/tf/dataset <br />

## CIFAR-10 (32x32 color image dataset)
* consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.<br />
* Download [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) <br />

## GTSRB (German Traffic Sign Recognition Benchmark dataset)
* Download [GTSRB_Final_Test_Images.zip](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip) <br/>
* Download [GTSRB_Final_Test_GT.zip](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip) <br />
* Download [GTSRB_Final_Training_Images.zip](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip) <br />

## FER2013 (Facial Expression Recognition dataset)
* Download & Unzip [FER2013.zip](https://anonfile.com/bdj3tfoeba/data_zip)<br />
*`$mv ~/Downloads/fer2013 ~/tf/dataset`* <br />

## Chest-Xray (Pneumonia Detection dataset)
* Download & Unzip [chest-xray-pneumonia.zip](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
) <br />
*`$mv ~/Downloads/chest-xray-pneumonia ~/tf/dataset`* <br />

## Garbage (Garbage Classification dataset)
* Download & unzip [Garbage Classification.zip](https://www.kaggle.com/asdasdasasdas/garbage-classification) <br />
*`$mv "Garbage Classification" ~/tf/dataset/garbage`* (rename to garbage) <br />

## nmt
* download [deu.txt](https://github.com/pradeepkannan85/Translation/blob/master/deu.txt) <br />
*`$mv ~/Downloads/deu.txt ~/tf/dataset/nmt`* <br />
* download [fra-eng.txt](https://github.com/L1aoXingyu/seq2seq-translation/blob/master/data/eng-fra.txt) <br />
*`$mv ~/Downloads/fra-eng.txt ~/tf/dataset/nmt`* <br />

## stock
* get API-Key free from https://www.alphavantage.co/ <br />
*`$pip install alpha_vantage`* <br />
*`$cd ~/tf`* <br />
*`$python download_stock_quote.py GOOGL`* <br />
* GOOGLE.csv saved into ~/tf/dataset/stock <br />
