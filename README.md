## Synopsis
The ImageNet challenge initiated by Fei-Fei Li (2010) has been traditionally approached
with image analysis algorithms such as SIFT with mitigated results until the late 90s.
The advent of Deep Learning dawned with a breakthrough in performance which was
gained by neural networks. Inspired by Yann LeCun et al. (1998) LeNet-5 model, the
first deep learning model, published by Alex Krizhevsky et al. (2012) drew attention to
the public by getting a top-5 error rate of 15.3% outperforming the previous best one
with an accuracy of 26.2% using a SIFT model. This model, the so-called 'AlexNet',
is what can be considered today as a simple architecture with five consecutive convolutional filters, max-pool layers, and three fully-connected layers. This project is designed to provide you with first-hand experience on training a typical Convolutional Neural Network (ConvNet) model in a discriminative classification
task. The model will be trained by Stochastic Gradient Descent, which is arguably the
canonical optimization algorithm in Deep Learning.

<b>Model</b><br/>
The ConvNet is a specific artificial neural network structure inspired by biological
visual cortex and tailored for computer vision tasks. The structure/architecture of a ConvNet is as follows.
1) Convolution. The convolution is the core building block of a ConvNet and
consists of a set of learnable filters. Every filter is a small receptive field. For
example, a typical filter on the first layer of a ConvNet might have size 5x5x3
(i.e., 5 pixels width and height, and 3 color channels).
2) Pooling. Pooling is a form of non-linear down-sampling. Max-pooling is the most
common. It partitions the input image into a set of non-overlapping rectangles
and, for each such sub-region, outputs the maximum.
3) Relu. Relu is non-linear activation function f(x) = max(0; x). It increases the
nonlinear properties of the decision function and of the overall network without
affecting the receptive fields of the convolution layer.

## Motivation
The advent of Deep Learning dawned with a breakthrough in performance which was
gained by neural networks. Inspired by Yann LeCun et al. (1998) LeNet-5 model, the
first deep learning model, published by Alex Krizhevsky et al. (2012) drew attention to
the public by getting a top-5 error rate of 15.3% outperforming the previous best one
with an accuracy of 26.2% using a SIFT model.

## Acknowledgements

This machine learning project is part of UCLA's Statistics 231/Computer Science 276A course on pattern recognition in machine learning, instructed by Professor Song-Chun Zhu.
