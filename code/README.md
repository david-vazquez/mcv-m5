# Keras implementation of Classification, Detection and Segmentation Networks

## Introduction

This repo contains the code to train and evaluate state of the art classification, detection and segmentation methods in a unified Keras framework working with Theano and/or TensorFlow. Pretrained models are also supplied.

## Available models

### Classification
 - [x] VGG16 and VGG19 network as described in [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf).
 
### Detection
 - [X] YOLO network as described in [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo.pdf).
  
### Segmentation
 - [x] FCN8 network as described in [Fully Convolutional Neural Networks](https://arxiv.org/abs/1608.06993).

## Available dataset wrappers

### Classification
 - [x] TT100K classsification dataset described in [Traffic-Sign Detection and Classification in the Wild](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf).
 
### Detection
 - [x] TT100K detection dataset described in [Traffic-Sign Detection and Classification in the Wild](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf).
  
### Segmentation
 - [x] Camvid dataset described in [Semantic Object Classes in Video: A High-Definition Ground Truth Database ](http://www.cs.ucl.ac.uk/staff/G.Brostow/papers/SemanticObjectClassesInVideo_BrostowEtAl2009.pdf).

## Installation
You need to install :
- [Theano](https://github.com/Theano/Theano) and [TensorFlow](https://github.com/Theano/Theano). Preferably the last version
- [Keras](https://github.com/fchollet/keras)

## Run experiments
All the parameters of the experiment are defined at config/dataset.py where dataset.py is the name of the dataset to use. Configure this file according to you needs.

To train/test a model in Theano, use the command: 

```
THEANO_FLAGS='device=cuda0,floatX=float32,lib.cnmem=0.95' python train.py -c config/dataset.py -e expName
```
 where dataset is the name of the dataset you want to use and expName the name of the experiment.

To train/test a model in TensorFlow, use the command: 

```
CUDA_VISIBLE_DEVICES=0 python train.py -c config/dataset.py -e expName
``` 
 where dataset is the name of the dataset you want to use and expName the name of the experiment.

All the logs of the experiments are stored in the result folder of the experiment.

## Authors
David Vázquez, Adriana Romero, Michal Drozdzal, Lluis Gomez

