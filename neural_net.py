# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    D, a hidden layer dimension of H, and performs classification over C classes.
    The network has the following architecture:
    input - fully connected layer - ReLU - fully connected layer - softmax
    The outputs of the second fully-connected layer are the scores for each class.
    """
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros((1, hidden_size))
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros((1, output_size))

    #
    def loss(self, X, y=None, reg=0.0):
        # 计算损失函数和梯度
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # the forward pass
        h1 = self.ReLU(np.dot(X, W1) + b1)   # b1 (1*H) 将其扩展为N*H
        scores = np.dot(h1, W2) + b2    # b2 (1*C) 将其扩展为H*C
        if y is None:
            return scores

        # the loss
        # N*1 每行是一个样本的最大分值
        scores_max = np.max(scores, axis=1, keepdims=True)
        # 取分值的对数，防止出现数据爆炸，因此减去scores_max  N*C
        exp_scores = np.exp(scores - scores_max)
        # 每行计算对数概率值  N*1
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # N*1 每个样本正确的类所得的对数概率
        correct_logprobs = -np.log(probs[range(N), y])
        # 所有样本的平均损失
        data_loss = np.sum(correct_logprobs) / N
        # 正则化损失  累加两个权值矩阵中所有权值的平方和
        reg_loss = 0.5 * reg * np.sum(W1*W1) + 0.5 * reg * np.sum(W2*W2)
        # 总体损失
        loss = data_loss + reg_loss

        # Backward pass: compute gradients
        grads = {}
        # Compute the gradient of scores
        dscores = probs                                 # (N,C)
        dscores[range(N), y] -= 1
        dscores /= N
        # Backprop into W2 and b2  展开所有式子可以推导得到
        dW2 = np.dot(h1.T, dscores)                     # (H,C)  累加了所有训练样本的梯度
        db2 = np.sum(dscores, axis=0, keepdims=True)    # (1,C)  累加所有输入
        # Backprop into hidden layer
        dh1 = np.dot(dscores, W2.T)                     # (N,H)
        # Backprop into ReLU non-linearity  大于0的才可以传播梯度
        dh1[h1 <= 0] = 0
        # Backprop into W1 and b1
        dW1 = np.dot(X.T, dh1)                          # (D,H)
        db1 = np.sum(dh1, axis=0, keepdims=True)        # (1,H)

        # Add the regularization gradient contribution
        dW2 += reg * W2
        dW1 += reg * W1
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads

    #
    def train(self, X, y, X_val, y_val, learning_rate=1e-3,
               learning_rate_decay=0.95, reg=1e-5, mu=0.9, num_epochs=10,
               mu_increase=1.0, batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
             X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
                               after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        # Use SGD to optimize the parameters
        v_W2, v_b2 = 0.0, 0.0
        v_W1, v_b1 = 0.0, 0.0
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(1, num_epochs * iterations_per_epoch + 1):
            X_batch = None
            y_batch = None
            # Sampling with replacement is faster than sampling without replacement.
            sample_index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[sample_index, :]        # (batch_size,D)
            y_batch = y[sample_index]           # (1,batch_size)

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # Perform parameter update (with momentum)
            v_W2 = mu * v_W2 - learning_rate * grads['W2']
            self.params['W2'] += v_W2
            v_b2 = mu * v_b2 - learning_rate * grads['b2']
            self.params['b2'] += v_b2
            v_W1 = mu * v_W1 - learning_rate * grads['W1']
            self.params['W1'] += v_W1
            v_b1 = mu * v_b1 - learning_rate * grads['b1']
            self.params['b1'] += v_b1
            """
            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
            """
            # Every epoch, check train and val accuracy and decay learning rate.
            if verbose and it % iterations_per_epoch == 0:
                # Check accuracy
                epoch = it / iterations_per_epoch
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                print 'epoch %d / %d: loss %f, train_acc: %f, val_acc: %f' %(epoch, num_epochs, loss, train_acc, val_acc)
                # Decay learning rate
                learning_rate *= learning_rate_decay
                # Increase mu
                mu *= mu_increase

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
             classify.
        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
                  the elements of X. For all i, y_pred[i] = c means that X[i] is
                  predicted to have class c, where 0 <= c < C.
        """
        y_pred = None
        h1 = self.ReLU(np.dot(X, self.params['W1']) + self.params['b1'])
        scores = np.dot(h1, self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)

        return y_pred

    def ReLU(self, x):
        """ReLU non-linearity."""
        return np.maximum(0, x)

