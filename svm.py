# -*- coding: utf-8 -*-

# - implement a fully-vectorized **loss function** for the SVM
# - implement the fully-vectorized expression for its **analytic gradient**
# - **check your implementation** using numerical gradient
# - use a validation set to **tune the learning rate and regularization** strength
# - **optimize** the loss function with **SGD**
# - **visualize** the final learned weights
#

import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt


#
def svm_loss_naive(W, X, y, reg):
    """
    # SVM 损失函数  native版本

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
    dW = np.zeros(W.shape)    # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    # 对于每一个样本，累加loss
    for i in xrange(num_train):
        scores = X[i].dot(W)     # (1, C)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            # 根据 SVM 损失函数计算
            margin = scores[j] - correct_class_score + 1    # note delta = 1
            # 当 margin>0 时，才会有损失，此时也会有梯度的累加
            if margin > 0:      # max(0, yi - yc + 1)
                loss += margin
                 # 根据公式：∇Wyi Li = - xiT(∑j≠yi1(xiWj - xiWyi +1>0)) + 2λWyi
                dW[:, y[i]] += -X[i, :]   # y[i] 是正确的类
                #  根据公式： ∇Wj Li = xiT 1(xiWj - xiWyi +1>0) + 2λWj ,
                dW[:, j] += X[i, :]

    # 训练数据平均损失
    loss /= num_train
    dW /= num_train

    # 正则损失
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    #
    return loss, dW


#
def svm_loss_vectorized(W, X, y, reg):
    """
    SVM 损失函数   向量化版本
    Structured SVM loss function, vectorized implementation.Inputs and outputs
    are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)   # initialize the gradient as zero
    scores = X.dot(W)        # N by C  样本数*类别数
    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores_correct = scores[np.arange(num_train), y]
    scores_correct = np.reshape(scores_correct, (num_train, 1))  # N*1 每个样本的正确类别

    margins = scores - scores_correct + 1.0     # N by C   计算scores矩阵中每一处的损失
    margins[np.arange(num_train), y] = 0.0      # 每个样本的正确类别损失置0
    margins[margins <= 0] = 0.0                 # max(0, x)
    loss += np.sum(margins) / num_train         # 累加所有损失，取平均
    loss += 0.5 * reg * np.sum(W * W)           # 正则

    # compute the gradient
    margins[margins > 0] = 1.0                  # max(0, x)  大于0的梯度计为1
    row_sum = np.sum(margins, axis=1)           # N*1  每个样本累加
    margins[np.arange(num_train), y] = -row_sum  # 类正确的位置 = -梯度累加
    dW += np.dot(X.T, margins)/num_train + reg * W     # D by C

    return loss, dW


#
def gradient_check(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
        检查 数值梯度 与 计算梯度 之间的差
        f: 损失计算函数
        x: 当前权值
        analytic_grad: 公式计算出的梯度
    """
    for i in xrange(num_checks):
        # 随机从权值矩阵 x 中选取一组下标
        ix = tuple([random.randrange(m) for m in x.shape])

        # 计算数值梯度
        oldval = x[ix]              # 该下标ix对应的权值
        x[ix] = oldval + h          # x+h
        fxph = f(x)                 # f(x + h)
        x[ix] = oldval - h          # x-h
        fxmh = f(x)                 # f(x - h)
        x[ix] = oldval              # reset
        grad_numerical = (fxph - fxmh) / (2 * h)

        # 计算差值
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical-grad_analytic)/(abs(grad_numerical)+abs(grad_analytic))
        print '数值梯度: %f 分析梯度: %f, 相对误差: %e' % (grad_numerical, grad_analytic, rel_error)


#
class svm_classifier:
    def __init__(self):
        pass

    def load_data(self):
        print "load cifar-10 data..."
        # Load the raw CIFAR-10 data.
        cifar10_dir = 'datasets/cifar-10'
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

        # As a sanity check, we print out the size of the training and test data.
        print 'Training data shape: ', X_train.shape
        print 'Training labels shape: ', y_train.shape
        print 'Test data shape: ', X_test.shape
        print 'Test labels shape: ', y_test.shape
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print "load data done...\n---------\n"

    #
    def visual_data(self):
        print "visualized some cifar-10 picture..."
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = len(classes)
        samples_per_class = 7
        for y, cl in enumerate(classes):
            idxs = np.flatnonzero(self.y_train == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(self.X_train[idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cl)
        plt.show()
        print "visualized data done...\n---------\n"

    #
    def pre_process_data(self):
        """
            分割 训练集、验证集、测试集
        """

        print "split training set and test set..."
        num_training = 49000
        num_validation = 1000
        num_test = 1000
        num_dev = 500

        # 验证集
        mask = range(num_training, num_training + num_validation)
        X_val = self.X_train[mask]
        y_val = self.y_train[mask]

        # 训练集
        mask = range(num_training)
        X_train = self.X_train[mask]
        y_train = self.y_train[mask]

        # dev集合为训练集的子集
        mask = np.random.choice(num_training, num_dev, replace=False)
        X_dev = self.X_train[mask]
        y_dev = self.y_train[mask]

        # 测试集
        mask = range(num_test)
        X_test = self.X_test[mask]
        y_test = self.y_test[mask]

        print 'Train data shape: ', X_train.shape
        print 'Train labels shape: ', y_train.shape
        print 'Validation data shape: ', X_val.shape
        print 'Validation labels shape: ', y_val.shape
        print 'Test data shape: ', X_test.shape
        print 'Test labels shape: ', y_test.shape

        # Preprocessing: reshape the image data into rows
        print "reshape the image data into rows..."
        X_train = np.reshape(X_train, (X_train.shape[0], -1))   # (49000, 3072)
        X_val = np.reshape(X_val, (X_val.shape[0], -1))         # (1000, 3072)
        X_test = np.reshape(X_test, (X_test.shape[0], -1))      # (1000, 3072)
        X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))         # (500, 3072)

        # As a sanity check, print out the shapes of the data
        print 'Training data shape: ', X_train.shape
        print 'Validation data shape: ', X_val.shape
        print 'Test data shape: ', X_test.shape
        print 'dev data shape: ', X_dev.shape

        print "均值归一化..."
        mean_image = np.mean(X_train, axis=0)   # 在训练集张每个像素的均值
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
        X_dev -= mean_image

        print "训练数据新增一列 全1 ，代表初始化的偏置..."
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])  # (49000, 3073)
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])        # (1000, 3073)
        X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])     # (1000, 3073)
        X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])        # (500, 3073)
        print X_train.shape, X_val.shape, X_test.shape, X_dev.shape

        #
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.X_dev, self.y_dev = X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

    #
    def train(self, X, y, learning_rate=1e-3, reg=1e-9, num_iters=1000,
                                            batch_size=100, verbose=True):
        """
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # 初始化权值
        print "-----"
        print "开始训练... \n 初始化权值... \n"
        W = np.random.randn(3073, 10) * 0.0001   # W:(D+1)*C 其中增加的一行            #代表偏置

        # 梯度检查
        # log:
        #忽略正则项:
            #数值梯度: -7.222939 分析梯度: -7.222939, 相对误差: 3.593999e-12
            #数值梯度: -4.257576 分析梯度: -4.257576, 相对误差: 4.057802e-12
            #数值梯度: 44.415218 分析梯度: 44.415218, 相对误差: 6.455885e-12
            #数值梯度: 2.262903 分析梯度: 2.262903, 相对误差: 9.628201e-12
            #数值梯度: 15.283688 分析梯度: 15.283688, 相对误差: 9.315487e-13
            #数值梯度: 26.637897 分析梯度: 26.637897, 相对误差: 3.047519e-14
            #数值梯度: -16.361032 分析梯度: -16.361032, 相对误差: 1.335984e-12
            #数值梯度: -1.476893 分析梯度: -1.476893, 相对误差: 6.077576e-12
            #数值梯度: 3.051969 分析梯度: 3.051969, 相对误差: 7.647957e-13
        #保留正则项:
            #数值梯度: -2.601181 分析梯度: -2.601181, 相对误差: 6.931185e-11
            #数值梯度: -7.508337 分析梯度: -7.508337, 相对误差: 7.611518e-12
            #数值梯度: 21.416632 分析梯度: 21.403071, 相对误差: 3.167032e-04
            #数值梯度: 8.034628 分析梯度: 8.034628, 相对误差: 3.358785e-11
            #数值梯度: 17.296864 分析梯度: 17.296864, 相对误差: 8.270697e-12
            #数值梯度: -19.669909 分析梯度: -19.669909, 相对误差: 1.966907e-11
            #数值梯度: -0.844342 分析梯度: -0.844342, 相对误差: 1.364827e-10
            #数值梯度: -34.325966 分析梯度: -34.325966, 相对误差: 4.523541e-12
            #数值梯度: 43.306574 分析梯度: 43.306574, 相对误差: 7.411165e-12
            #数值梯度: 27.153273 分析梯度: 27.153273, 相对误差: 8.958236e-12

        flag_grad_check = False
        if flag_grad_check:
            print "进行梯度检查...\n"
            # 梯度检查 without Regularization
            print "  忽略正则项: "
            loss, grad = svm_loss_vectorized(W, self.X_dev, self.y_dev, 0.0)
            f = lambda w: svm_loss_vectorized(W, self.X_dev, self.y_dev, 0.0)[0]  # 计算返回损失
            grad_numerical = gradient_check(f, W, grad)

            # 梯度检查 with Regularization
            print "  保留正则项: "
            loss, grad = svm_loss_vectorized(W, self.X_dev, self.y_dev, 1e2)
            f = lambda w: svm_loss_vectorized(W, self.X_dev, self.y_dev, 1e2)[0]
            grad_numerical = gradient_check(f, W, grad)

        ###############
        # 训练
        num_train, dim = X.shape
        num_classes = np.max(y) + 1   # assume y takes values 0...K-1 where K is number of classes
        if W is None:
            # lazily initialize W
            W = 0.001 * np.random.randn(dim, num_classes)

        # 随机梯度下降
        loss_history = []
        for it in xrange(num_iters):
            choice_idx = np.random.choice(num_train, batch_size)
            X_batch = X[choice_idx]
            y_batch = y[choice_idx]

            # evaluate loss and gradient
            loss, grad = svm_loss_vectorized(W, X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            W += -learning_rate * grad

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
            if loss <= 0.1:
                print "early stopping..."
                break

        self.W = W
        # 日志：
        #iteration 0 / 200: loss 8.706914
        #iteration 10 / 200: loss 1030.248971
        #iteration 20 / 200: loss 697.499035
        #iteration 30 / 200: loss 969.223779
        #iteration 40 / 200: loss 154.672770
        #iteration 50 / 200: loss 43.691685
        #iteration 60 / 200: loss 232.523850
        #iteration 70 / 200: loss 15.329242
        #iteration 80 / 200: loss 3.348309
        #iteration 90 / 200: loss 20.153373
        #iteration 100 / 200: loss 1.350941
        #iteration 110 / 200: loss 0.530806
        #iteration 120 / 200: loss 0.249542
        #iteration 130 / 200: loss 0.479288
        #iteration 140 / 200: loss 0.009414
        #iteration 150 / 200: loss 0.287484
        #iteration 160 / 200: loss 0.204978
        #iteration 170 / 200: loss 0.134885
        #iteration 180 / 200: loss 0.000180
        #iteration 190 / 200: loss 0.000180

        # test the accuracy rate
        acc_train = self.predict(self.X_dev, self.y_dev)
        acc_valid = self.predict(self.X_val, self.y_val)
        print "acc_train = ", acc_train, " acc_valid = ", acc_valid
        return acc_train, acc_valid

    # 预测
    def predict(self, X, y):
        #print "----\n 预测...  \nX.shape = ", X.shape
        #print "y.shape = ", y.shape
        #print "W.shape = ", self.W.shape
        # X: N*D
        y_pred = np.zeros(X.shape[0])    # 1 by N
        y_pred = np.argmax(np.dot(X, self.W), axis=1)

        acc = np.mean(y == y_pred)
        #print "准确率是: ", acc
        return acc

    #
    def cross_valid(self):
        """
        在验证集上选取最优的
        """
        learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
        regularization_strengths = [1e2, 1.0, 1e-1, 1e-2, 1e-3]
        results = {}
        best_para = (-1, -1)
        best_val = -float("inf")
        iters = 1500
        for lr in learning_rates:
            for rs in regularization_strengths:
                print "\n\n*******"
                print "lr = ", lr, " reg = ", rs
                acc_train, acc_valid = self.train(self.X_dev, self.y_dev, learning_rate=lr, reg=rs, num_iters=1000, batch_size=100, verbose=True)
                results[(lr, rs)] = (acc_train, acc_valid)
                if best_val < acc_valid:
                    best_val = acc_valid
                    best_para = (lr, rs)

        # print results
        for lr, reg in sorted(results):
            train_accuracy, val_accuracy = results[(lr, reg)]
            print 'lr %e reg %e train accuracy: %f val accuracy: %f' %(lr, reg, train_accuracy, val_accuracy)
        print 'Best validation accuracy achieved during validation: %f' % best_val # about 38.2%

    #
    def visual_weight(self):
        lr = 1e-2
        rs = 0

        acc_train, acc_valid = self.train(self.X_train[:10000], self.y_train[:10000], learning_rate=lr, reg=rs, num_iters=100000, batch_size=1000, verbose=True)

        w = self.W[:-1, :]   # strip out the bias
        w = w.reshape(32, 32, 3, 10)
        w_min, w_max = np.min(w), np.max(w)
        classes = ['plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

        for i in xrange(10):
            plt.subplot(2, 5, i + 1)
            # 规约到0-255
            wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
            plt.imshow(wimg.astype('uint8'))
            plt.axis('off')
            plt.title(classes[i])
        #plt.show()
        plt.savefig("weight_svm.jpg")


#
if __name__ == '__main__':
    svm = svm_classifier()
    # 载入数据
    svm.load_data()
    # 可视化部分数据
    #svm.visual_data()
    # 预处理数据
    svm.pre_process_data()
    #
    #loss_hist = svm.train(svm.X_dev, svm.y_dev)
    #
    plot_loss = False
    if plot_loss:
        plt.plot(loss_hist)
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        plt.show()

    # 在验证集上选取最优的权值
    #svm.cross_valid()
    # 可视化权值
    svm.visual_weight()


"""
# Visualize the cross-validation results
import math
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results] # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()


# In[ ]:

# Evaluate the best svm on test set
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print 'linear SVM on raw pixels final test set accuracy: %f' % test_accuracy


# In[ ]:

# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
w = best_svm.W[:-1,:] # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in xrange(10):
  plt.subplot(2, 5, i + 1)

  # Rescale the weights to be between 0 and 255
  wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
  plt.imshow(wimg.astype('uint8'))
  plt.axis('off')
  plt.title(classes[i])


# ### Inline question 2:
# Describe what your visualized SVM weights look like, and offer a brief explanation for why they look they way that they do.
#
# **Your answer:** *fill this in*
"""
