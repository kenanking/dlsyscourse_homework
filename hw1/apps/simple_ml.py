import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, 'rb') as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows*cols)
        # normalize data to [0.0, 1.0]
        X = X.astype(np.float32) / 255.0
        
    with gzip.open(label_filename, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)
    
    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    N, K = Z.shape
    Z_exp = ndl.exp(Z)
    Z_exp_sum = ndl.summation(Z_exp, axes=(1,))
    Z_exp_sum = ndl.reshape(Z_exp_sum, (N, 1))
    Z_exp_sum = ndl.broadcast_to(Z_exp_sum, Z_exp.shape)
    Z_exp_norm = Z_exp / Z_exp_sum
    
    loss = -ndl.summation(y_one_hot * ndl.log(Z_exp_norm)) / N
    return loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    N, D = X.shape
    _, H = W1.shape
    _, K = W2.shape

    for i in range(0, N, batch):
        print(f"batch {i} / {N}")
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]
        y_one_hot = np.zeros((len(y_batch), K))
        y_one_hot[np.arange(len(y_batch)), y_batch] = 1

        X_batch = ndl.Tensor(X_batch)
        y_one_hot = ndl.Tensor(y_one_hot)

        Z1 = ndl.relu(ndl.matmul(X_batch, W1))
        Z2 = ndl.matmul(Z1, W2)
        loss = softmax_loss(Z2, y_one_hot)

        loss.backward()
        W1 -= lr * W1.grad
        W2 -= lr * W2.grad

        W1 = ndl.Tensor.make_const(W1, requires_grad=True)
        W2 = ndl.Tensor.make_const(W2, requires_grad=True)

    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
