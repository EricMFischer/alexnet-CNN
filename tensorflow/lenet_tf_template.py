import tensorflow as tf
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
import math
import matplotlib.pyplot as plt
from PIL import Image

USE_GPU = True

###################################################################################################
#                                            PREAMBLE                                             #
###################################################################################################

log = lambda *args: print(datetime.now().strftime('%H:%M:%S'), ':', *args)

def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10

    X_train = np.asarray(X_train, dtype=np.float32)
    print('X_train: ', len(X_train))  # 50000 examples, 32x32x3
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    print('y_train: ', y_train)  # vector of 50000 ints 0-9 for img classes

    X_test = np.asarray(X_test, dtype=np.float32)
    print('X_test: ', len(X_test))  # 10000 examples, 32x32x3
    y_test = np.asarray(y_test, dtype=np.int32).flatten()
    print('y_test: ', y_test)  #  vector of 10000 ints 0-9

    # validation examples
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    print('X_val: ', len(X_val))  # 1000 examples, 32x32x3
    y_val = y_train[mask]
    print('y_val: ', len(y_val))  # vector of 1000 ints 0-9

    # training examples
    mask = range(num_training)
    X_train = X_train[mask]  # 49000 examples, 32x32x3
    y_train = y_train[mask]  # vector of 49000 ints 0-9

    # testing examples
    mask = range(num_test)
    X_test = X_test[mask]  # 10000 examples, 32x32x3
    y_test = y_test[mask]  # vector of 10000 ints 0-9

    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    print('mean pixel intensity: ', mean_pixel)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    print('standard deviation of pixel intensity: ', std_pixel)


    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test, mean_pixel, std_pixel


class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):  # batch_size = 64
        assert X.shape[0] == y.shape[0]
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i + B], self.y[i:i + B]) for i in range(0, N, B))


X_train, y_train, X_val, y_val, X_test, y_test, mean_pixel, std_pixel = load_cifar10()
train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Dataset(X_test, y_test, batch_size=64)

get_X_train_sample = lambda: next(iter(train_dset))[0][7] * std_pixel + mean_pixel

def select_device(use_gpu=True):
    from tensorflow.python.client import device_lib
    log(device_lib.list_local_devices())
    device = '/device:GPU:0' if use_gpu else '/CPU:0'
    log('Using device: ', device)
    return device

device = select_device(use_gpu=USE_GPU)

###################################################################################################
#                                              PART 1                                             #
###################################################################################################


def flatten(x):
    """
    Input:
    - TensorFlow Tensor of shape (N, D1, ..., DM)

    Output:
    - TensorFlow Tensor of shape (N, D1 * ... * DM)
    """
    x_flat = None
    ############################################################################
    # TODO: (1.a) Reshape tensor x into shape (N, D1 * ... * DM)               #
    ############################################################################

    x_flat = tf.layers.flatten(x)

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
    return x_flat


def kaiming_normal(shape):
    """
    He et al, *Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification, ICCV 2015, https://arxiv.org/abs/1502.01852
    """
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]
    # tf.random_normal: 1-d tensor shape input/output, outputs rand values from norm dist
    return tf.random_normal(shape) * np.sqrt(2.0 / fan_in)


def convnet_init():
    """
    Initialize the weights of a Three-Layer ConvNet, for use with the
    three_layer_convnet function defined above.
    """

    '''
    conv_w1 = tf.Variable(kaiming_normal([5, 5, 3, 32]))  # 5x5x3 filter * num of filters
    conv_b1 = tf.Variable(tf.zeros(32,))
    print('conv_w1: ', conv_w1)  # 5x5x3x32 tensor

    conv_w2 = tf.Variable(kaiming_normal([5, 5, 32, 32]))  # shape 5x5; 32 input and output channels
    conv_b2 = tf.Variable(tf.zeros(32,))

    conv_w3 = tf.Variable(kaiming_normal([5, 5, 32, 64]))
    conv_b3 = tf.Variable(tf.zeros(64,))

    ###########################################################################
    # TODO: (1.a), (2.a) Initialize the remaining parameters.                 #
    ###########################################################################

    conv_w4 = tf.Variable(kaiming_normal([4, 4, 64, 64]))
    conv_b4 = tf.Variable(tf.zeros(64,))

    conv_w5 = tf.Variable(kaiming_normal([1, 1, 64, 10]))
    conv_b5 = tf.Variable(tf.zeros(10,))
    '''

    # For a convolution with input volume size W, filter size F, stride S, padding P, the output volume size is (W-F+2P)/S+1.


    # ConvNet with only Block1 and Block5
    # adjust filter size in block 5 accordingly
#     conv_w1 = tf.Variable(kaiming_normal([5, 5, 3, 32]))  # 5x5x3 filter * num of filters
#     conv_b1 = tf.Variable(tf.zeros(32,))
#     conv_w5 = tf.Variable(kaiming_normal([1, 1, 32, 10]))
#     conv_b5 = tf.Variable(tf.zeros(10,))
#     params = [conv_w1, conv_b1, conv_w5, conv_b5]


    # ConvNet with only Block1, Block2, and Block5
#     conv_w1 = tf.Variable(kaiming_normal([5, 5, 3, 32]))  # 5x5x3 filter * num of filters
#     conv_b1 = tf.Variable(tf.zeros(32,))
#     conv_w2 = tf.Variable(kaiming_normal([5, 5, 32, 32]))  # shape 5x5; 32 input and output channels
#     conv_b2 = tf.Variable(tf.zeros(32,))
#     conv_w5 = tf.Variable(kaiming_normal([1, 1, 32, 10]))
#     conv_b5 = tf.Variable(tf.zeros(10,))
#     params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w5, conv_b5]

    # ConvNet with only Block1, Block2, Block3, and Block5
    conv_w1 = tf.Variable(kaiming_normal([5, 5, 3, 32]))  # 5x5x3 filter * num of filters
    conv_b1 = tf.Variable(tf.zeros(32,))
    conv_w2 = tf.Variable(kaiming_normal([5, 5, 32, 32]))  # shape 5x5; 32 input and output channels
    conv_b2 = tf.Variable(tf.zeros(32,))
    conv_w3 = tf.Variable(kaiming_normal([5, 5, 32, 64]))
    conv_b3 = tf.Variable(tf.zeros(64,))
    conv_w5 = tf.Variable(kaiming_normal([1, 1, 64, 10]))
    conv_b5 = tf.Variable(tf.zeros(10,))
    params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w5, conv_b5]


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # list of tensorflow tensors holding the randomly initialized weights of the model
    # params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w4, conv_b4, conv_w5, conv_b5]

    return params


def convnet_forward(x, params):
    """
    A three-layer convolutional network.

    Args:
    - x: A TensorFlow Tensor of shape (N, H, W, 3) (num samples, height, width, channels) giving a minibatch of images
    - params: A list of TensorFlow Tensors giving the weights and biases for the network

    Output:
    - TensorFlow Tensor of shape (N, C) giving scores for all elements of x.

    """
    # [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w4, conv_b4, conv_w5, conv_b5] = params
    # ConvNet with only Block1 and Block5:
    # [conv_w1, conv_b1, conv_w5, conv_b5] = params
    # ConvNet with only Block1, Block2, and Block5:
    # [conv_w1, conv_b1, conv_w2, conv_b2, conv_w5, conv_b5] = params
    # ConvNet with only Block1, Block2, Block3, and Block5:
    [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w5, conv_b5] = params



    # block 1
    # pad(input tensor (n=rank,2), before/after paddings for each dimension of input, constant mode, pad with zeros)
    x1_1_pad = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', constant_values=0)
    '''
    conv(input tensor [batch, in_height, in_width, in_channels], filter tensor [filter_height, filter_width, in_channels, out_channels], strides, padding)
    Computes a 2-D convolution given 4-D input and filter tensors.
    1) Flattens the filter to a 2-D matrix with shape [filter_height*filter_width*in_channels, output_channels].
    2) Extracts image patches from the input tensor to form a virtual tensor of shape [batch, out_height, out_width, filter_height*filter_width*in_channels].
    3) For each patch, right-multiplies the filter matrix and the image patch vector.
    '''
    x1_2_conv = tf.nn.conv2d(x1_1_pad, conv_w1, [1, 1, 1, 1], padding='VALID') + conv_b1
    x1_3_pad = tf.pad(x1_2_conv, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    '''
    max_pool arguments:
    value: A 4-D Tensor of the format specified by data_format.
    ksize: A list or tuple of 4 ints. The size of the window for each dimension of the input tensor.
    strides: A list or tuple of 4 ints. The stride of the sliding window for each dimension of the input tensor.
    '''
    x1_4_pool = tf.nn.max_pool(x1_3_pad, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    x1_5_relu = tf.nn.relu(x1_4_pool)  # Computes rectified linear: max(features, 0)

    # printing output of block 1
    # print('Size after conv1: ', x1_5_relu.get_shape())

    # block 2
    x2_1_pad = tf.pad(x1_5_relu, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', constant_values=0)
    x2_2_conv = tf.nn.conv2d(x2_1_pad, conv_w2, [1, 1, 1, 1], padding='VALID') + conv_b2
    x2_3_relu = tf.nn.relu(x2_2_conv)
    x2_4_pad = tf.pad(x2_3_relu, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    x2_5_pool = tf.nn.avg_pool(x2_4_pad, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')  # avg pooling of input

    # block 3
    x3_1_pad = tf.pad(x2_5_pool, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', constant_values=0)
    x3_2_conv = tf.nn.conv2d(x3_1_pad, conv_w3, [1, 1, 1, 1], padding='VALID') + conv_b3
    x3_3_relu = tf.nn.relu(x3_2_conv)
    x3_4_pad = tf.pad(x3_3_relu, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    x3_5_pool = tf.nn.avg_pool(x3_4_pad, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    ############################################################################
    # TODO: (1.a), (2.a) Implement the remaining forward pass.                 #
    ############################################################################

    # block 4
#     x4_1_pad = tf.pad(x3_5_pool, [[0, 0], [0, 0], [0, 0], [0, 0]], mode='CONSTANT', constant_values=0)
#     x4_2_conv = tf.nn.conv2d(x4_1_pad, conv_w4, [1, 1, 1, 1], padding='VALID') + conv_b4
#     x4_3_relu = tf.nn.relu(x4_2_conv)

    # block 5
#     x5_1_pad = tf.pad(x4_3_relu, [[0, 0], [0, 0], [0, 0], [0, 0]], mode='CONSTANT', constant_values=0)
#     x5_2_conv = tf.nn.conv2d(x5_1_pad, conv_w5, [1, 1, 1, 1], padding='VALID') + conv_b5

    # ConvNet with only Block1 and Block5:
    x5_1_pad = tf.pad(x3_5_pool, [[0, 0], [0, 0], [0, 0], [0, 0]], mode='CONSTANT', constant_values=0)
    x5_2_conv = tf.nn.conv2d(x5_1_pad, conv_w5, [1, 1, 1, 1], padding='VALID') + conv_b5
    # ConvNet with only Block1, Block2, and Block5:
#     x5_1_pad = tf.pad(x2_5_pool, [[0, 0], [0, 0], [0, 0], [0, 0]], mode='CONSTANT', constant_values=0)
#     x5_2_conv = tf.nn.conv2d(x5_1_pad, conv_w5, [1, 1, 1, 1], padding='VALID') + conv_b5
    # ConvNet with only Block1, Block2, Block3, and Block5:
#     x5_1_pad = tf.pad(x3_5_pool, [[0, 0], [0, 0], [0, 0], [0, 0]], mode='CONSTANT', constant_values=0)
#     x5_2_conv = tf.nn.conv2d(x5_1_pad, conv_w5, [1, 1, 1, 1], padding='VALID') + conv_b5

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################

    logits = flatten(x5_2_conv)

    return logits


def three_layer_convnet_test():
    tf.reset_default_graph()  # do not call while a tf.Session is active

    with tf.device(device):
        # tf.placeholder: tensor will produce an error if evaluated. Its value must
        # be fed w/ the feed_dict opt. arg. to Session.run(), Tensor.eval(), or Operation.run()
        x = tf.placeholder(tf.float32)

        # block 1
        conv_w1 = tf.zeros([5, 5, 3, 32])
        conv_b1 = tf.zeros(32)

        # block 2
        conv_w2 = tf.zeros([5, 5, 32, 32])
        conv_b2 = tf.zeros(32)

        # block 3
        conv_w3 = tf.zeros([5, 5, 32, 64])
        conv_b3 = tf.zeros(64)

        ############################################################################
        # TODO: (1.a), (2.a) Initialize the parameters.                            #
        ############################################################################

        # block 4
#         conv_w4 = tf.zeros([4, 4, 64, 64])
#         conv_b4 = tf.zeros(64)

        # block 5
        conv_w5 = tf.zeros([1, 1, 64, 10])
        # conv_w5 = tf.zeros([1, 1, 32, 10])
        conv_b5 = tf.zeros(10)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w4, conv_b4, conv_w5, conv_b5]
        # ConvNet with only Block1 and Block5:
        # params = [conv_w1, conv_b1, conv_w5, conv_b5]
        # ConvNet with only Block1, Block2, and Block5:
        # params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w5, conv_b5]
        # ConvNet with only Block1, Block2, Block3, and Block5:
        params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w5, conv_b5]

        logits = convnet_forward(x, params)
        print('params: ', params)
        print('logits: ', logits)

    # Inputs to convolutional layers are 4-dimensional arrays with shape [batch_size, height, width, channels]
    x_np = np.zeros((64, 32, 32, 3))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # x was a placeholder until fed data here during session run
        logits_np = sess.run(logits, feed_dict={x: x_np})
        log('logits_np has shape', format(logits_np.shape))


with tf.device('/cpu:0'):
    three_layer_convnet_test()

def training_step(logits, y, params, learning_rate):
    """
    Set up the part of the computational graph which makes a training step.

    Args:
    - logits: TensorFlow Tensor of shape (N, C) giving classification scores for
      the model.
    - y: TensorFlow Tensor of shape (N,) giving ground-truth labels for scores;
      y[i] == c means that c is the correct class for scores[i].
    - params: List of TensorFlow Tensors giving the weights of the model
    - learning_rate: Python scalar giving the learning rate to use for gradient
      descent step.

    Returns:
    - loss: A TensorFlow Tensor of shape () (scalar) giving the loss for this
      batch of data; evaluating the loss also performs a gradient descent step
      on params (see above).
    """
    # First compute the loss; the first line gives losses for each example in
    # the mini-batch, and the second averages the losses across the batch
    '''
    sparse_softmax_cross_entropy:
    Measures the probability error in discrete classification tasks in which the classes are mutually exclusive.
    returns: tensor same shape as labels and same type as logits with the softmax cross entropy loss.
    '''
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    print('losses: ', losses)
    # tf.reduce_mean: computes mean of elems across dimensions of tensor (2nd arg -> axis)
    loss = tf.reduce_mean(losses)
    print('loss: ', loss)

    # Compute the gradient of the loss with respect to each parameter of the the
    # network. This is a very magical function call: TensorFlow internally
    # traverses the computational graph starting at loss backward to each element
    # of params, and uses back-propagation to figure out how to compute gradients;
    # it then adds new operations to the computational graph which compute the
    # requested gradients, and returns a list of TensorFlow Tensors that will
    # contain the requested gradients when evaluated.
    grad_params = tf.gradients(loss, params)
    print('grad_params: ', grad_params)  # array with 10 tensors same shapes as our params

    # Make a gradient descent step on all of the model parameters.
    new_weights = []
    for w, grad_w in zip(params, grad_params):
        new_w = tf.assign_sub(w, learning_rate * grad_w)
        new_weights.append(new_w)

    # Insert a control dependency so that evaluating the loss causes a weight
    # update to happen.
    # control_dependencies: input -> list of operation or tensor objects which must be
    # executed or computed before running the operations defined in the context.
    with tf.control_dependencies(new_weights):
        return tf.identity(loss)


def train(model_fn, init_fn, learning_rate, epochs, print_every=100):
    """
    Train a model on CIFAR-10.

    Args:
    - model_fn: A Python function that performs the forward pass of the model
      using TensorFlow; it should have the following signature:
      scores = model_fn(x, params) where x is a TensorFlow Tensor giving a
      minibatch of image data, params is a list of TensorFlow Tensors holding
      the model weights, and scores is a TensorFlow Tensor of shape (N, C)
      giving scores for all elements of x.
    - init_fn: A Python function that initializes the parameters of the model.
      It should have the signature params = init_fn() where params is a list
      of TensorFlow Tensors holding the (randomly initialized) weights of the
      model.
    - learning_rate: Python float giving the learning rate to use for SGD.
    """
    # First clear the default graph
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, name='is_training')
    # Set up the computational graph for performing forward and backward passes,
    # and weight updates.
    with tf.device(device):
        # Set up placeholders for the data and labels
        # Passing None to a shape argument of a tf.placeholder tells it simply that
        # that dimension is unspecified, and to infer that dimension from the tensor
        # you are feeding it during run-time (when you run a session). Only some arguments
        # (generally the batch_size argument) can be set to None since Tensorflow needs to
        # be able to construct a working graph before run time. This is useful for
        # when you don't want to specify a batch_size before run time.
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None]) # will be a vector of 64
        params = init_fn()  # Initialize the model parameters
        scores = model_fn(x, params)  # Forward pass of the model
        loss = training_step(scores, y, params, learning_rate)

    train_losses = []
    test_accuracies = []

    # Now we actually run the graph many times using the training data
    with tf.Session() as sess:
        # Initialize variables that will live in the graph
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            log('epoch {:>4d}/{:>4d}'.format(epoch, epochs))
            epoch_time = time()
            for t, (x_np, y_np) in enumerate(train_dset):
                # Run the graph on a batch of training data; recall that asking
                # TensorFlow to evaluate loss will cause an SGD step to happen.
                feed_dict = {x: x_np, y: y_np}  # y_np is a vector of 64
                loss_np = sess.run(loss, feed_dict=feed_dict)

                # Periodically print the loss and check accuracy on the val set
                if t % print_every == 0:
                    num_correct, num_samples, acc = get_accuracy(sess, val_dset, x, scores, is_training)
                    log('   iteration = {:>4d}, loss = {:>8.4f}, accuracy = {:>8.2f}%'.format(t, loss_np, acc))

            train_losses.append(loss_np)
            test_accuracies.append(acc)

            log('epoch {:>4d} took {:>.2f}s'.format(epoch, time()-epoch_time))

        return params, sess.run(params), train_losses, test_accuracies


def get_accuracy(sess, dset, x, logits, is_training=None):
    """
    Check accuracy on a classification model.

    Args:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.

    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(logits, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    return num_correct, num_samples, 100 * acc



############################################################################
# TODO: (1.b) Adjust learning-rate and number of epochs.                   #
############################################################################
learning_rate = 0.05
epochs = 10
############################################################################
#                             END OF YOUR CODE                             #
############################################################################

params, params_val, train_losses, test_accuracies = train(convnet_forward, convnet_init, learning_rate, epochs=epochs)
print('params: ', params)
print('test_accuracies: ', test_accuracies)


############################################################################
# TODO: (1.c) Plot.                                                        #
############################################################################

plt.plot(train_losses)
plt.ylabel('Training Losses')
plt.xlabel('Epochs')
plt.show()
plt.savefig('1_loss.png')


plt.plot(test_accuracies)
plt.ylabel('Test Accuracies')
plt.xlabel('Epochs')
plt.show()
plt.savefig('2_accuracy.png')

pass

############################################################################
#                             END OF YOUR CODE                             #
############################################################################


###################################################################################################
#                                              PART 2                                             #
###################################################################################################


def plot_kernels_on_grid(kernel, grid_Y, grid_X, pad = 1):
    """
    Visualize convolutional features as an image.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    """

    x_min = tf.reduce_min(kernel)  # min of elements across dimensions of tensor, axis=None so all dimensions are reduced to return single elem
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    # padding: starts with rows above/below, then columns before/after, then for 3rd and 4th dimensions NumChannels and NumKernels it adds nothing
    # Saves value to x1
    x1 = tf.pad(kernel1, tf.constant([[pad,pad], [pad, pad], [0,0], [0,0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad  # saves Y dimension, considering padding added
    X = kernel1.get_shape()[1] + 2 * pad  # saves X dimension, considering padding added

    channels = kernel1.get_shape()[2]  # NumChannels

    # put NumKernels to the 1st dimension
    # The returned tensor's dimension i will correspond to the input dimension perm[i].
    # If perm is not given, it is set to (n-1...0), where n is the rank of the input tensor.
    # Hence by default, this operation performs a regular matrix transpose on 2-D input Tensors.
    x2 = tf.transpose(x1, (3, 0, 1, 2))  # transposes x1, permutes dimensions according to 2nd arg
    # organize grid on Y axis
    # tf.reshape: Given tensor, this operation returns a tensor that has the same values as tensor with shape shape.
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))  # stacked shape(4,)

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels], where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


grid = plot_kernels_on_grid(params[0], 4, 8)

with tf.Session() as sess:
    ############################################################################
    # TODO: Retrieve image of kernels from symbolic 'grid' variable.           #
    ############################################################################

    # grid_val = [None]
    sess.run(tf.global_variables_initializer())
    grid_val = sess.run(grid)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    img = Image.fromarray(grid_val[0], 'RGB')
    img.save('3_kernels.jpeg')

    # 2b) Compare the final test accuraries for (i., ii., iii.) in a Table.
    fig, ax = plt.subplots()
    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    acc_data = [[17.2, 18.8, 19.5, 19.2, 19.8, 20.3, 20.7, 20.7, 21.2, 21.6],
                [20.4, 25.3, 25.3, 26.700000000000003, 27.6, 28.499999999999996, 28.499999999999996, 28.7, 28.999999999999996, 31.7],
                [35.199999999999996, 43.4, 47.099999999999994, 50.1, 52.800000000000004, 54.0, 55.1, 56.49999999999999, 56.799999999999996, 56.99]]
    clust_data = np.transpose(acc_data)
    collabel=("Blocks 1,5", "Blocks 1,2,5", "Blocks 1,2,3,5")
    ax.table(cellText=clust_data,colLabels=collabel,loc='center')
    plt.show()


###################################################################################################
#                                              PART 3                                             #
###################################################################################################


def plot_filter_grid(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 12))
    n_columns = 8
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.axis('off')
        plt.imshow(units[0, :, :, i], interpolation='nearest')


def conv1_activations(x, conv_w1, conv_b1):
    ############################################################################
    # TODO: Compute activations for the first conv layer.                      #
    ############################################################################
    print('conv_w1: ', conv_w1)
    print('conv_b1: ', conv_b1)
    x1_1_pad = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', constant_values=0)
    x1_2_conv = tf.nn.conv2d(x1_1_pad, conv_w1, [1, 1, 1, 1], padding='VALID') + conv_b1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return x1_2_conv


image = get_X_train_sample()
plt.imshow(np.squeeze(image).astype(np.uint8), interpolation='nearest')
plt.figure(1, figsize=(10, 10))
plt.axis('off')
plt.savefig('4_data.png', bbox_inches='tight')
plt.show()

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    result = flatten(x)
    conv1_a = conv1_activations(x, params[0], params[1])
    # Plot the learned 32 filters of the first convolutional layer in LeNet.
    print('conv1_a: ', conv1_a)
    hidden_1 = sess.run(conv1_a, feed_dict={x: image, params[0]: params_val[0], params[1]: params_val[1]})
    plot_filter_grid(hidden_1)
    plt.savefig('5_activations.png', bbox_inches='tight')
    # Plot the filter response maps for a given sample image of CIFAR-10.
    plt.show()
