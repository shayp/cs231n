import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  dW = np.zeros_like(W)
  numTrain = y.shape[0]
  numClasses = W.shape[1]
  inputSize = X.shape[1]

  loss = 0.0

  # run for all training set
  for i in xrange(numTrain):
    # computer W* X[i]
    score = W.T.dot(X[i])

    # compote softmax with numeric stability
    softmax = np.exp(score - np.max(score))
    softmax /= np.sum(softmax)

    # softmax log likelihood : -log(pi)
    log_likelihood = -np.log(softmax[y[i]])

    # add to total loss
    loss += np.sum(log_likelihood)

    # calculate q for gradient
    q = np.reshape(softmax, (numClasses, 1))

    # sub from the real yi 1 to get (1 - pi)
    q[y[i]] -= 1
    currExample = np.reshape(X[i], (inputSize, 1))

    # change loss with respect to current example
    dW += currExample.dot(q.T)

  # normalize to train set size
  dW /= numTrain
  loss /= numTrain

  # add regularization penalty to loss and gradient
  loss += reg * np.sum(W * W)
  dW += -2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  numTrain = y.shape[0]

  # compute scores
  score = X.dot(W)

  # compute softmax with numeric stability
  softmax = np.exp(score - np.max(score, axis=1, keepdims=True))
  softmax /= np.sum(softmax, axis=1, keepdims=True)

  # calculate log likelihood : -sum(log(pi))
  log_likelihood = -np.log(softmax[range(numTrain), y])
  loss += np.sum(log_likelihood)

  # calculate gradient
  ind = np.zeros_like(softmax)
  ind[np.arange(numTrain), y] = 1
  Q = softmax - ind
  dW = X.T.dot(Q)

  # normalize
  dW /= numTrain
  loss /= numTrain

  # add L2 regularization
  loss += reg * np.sum(W * W)
  dW += -2 * reg * W

  return loss, dW

