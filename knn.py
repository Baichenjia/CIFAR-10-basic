# -*- coding: utf-8 -*-
__author__ = "Bai Chenjia"

# The kNN classifier consists of two stages:
# During training, the classifier takes the training data and simply remembers it
# During testing, kNN classifies every test image by comparing to all training images and transfering the labels of the k most similar training examples
# The value of k is cross-validated
#
# In this exercise you will implement these steps and understand the basic Image Classification pipeline, cross-validation, and gain proficiency in writing efficient, vectorized code.


import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import os
import time

#u'pgf', u'cairo', u'MacOSX', u'CocoaAgg', u'gdk', u'ps', u'GTKAgg', u'nbAgg', u'GTK', u'Qt5Agg', u'template', u'emf', u'GTK3Cairo', u'GTK3Agg', u'WX', u'Qt4Agg', u'TkAgg', u'agg', u'svg', u'GTKCairo', u'WXAgg', u'WebAgg', u'pdf']


class KNN_classifier:
    """
        用KNN分类器在 CIFAR-10 图像上进行图像分类。
        该方法在实际的图像分类中不可用，仅作为练习
    """
    def __init__(self):
        """
            初始化
        """
        plt.rcParams['figure.figsize'] = (10.0, 8.0)
        #plt.rcParams['image.interpolation'] = 'nearest'
        #plt.rcParams['image.cmap'] = 'gray'
        print "----"

    def load_CIFAR10(self):
        """
            读取数据并抽样可视化
            Load the raw CIFAR-10 data.
            show a few examples of training images from each class.
        """
        cifar10_dir = 'datasets/cifar-10'
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

        # As a sanity check, we print out the size of the training and test data.
        print 'Training data shape: ', X_train.shape     # (50000, 32, 32, 3)
        print 'Training labels shape: ', y_train.shape   # (50000,)
        print 'Test data shape: ', X_test.shape          # (10000, 32, 32, 3)
        print 'Test labels shape: ', y_test.shape        # (10000,)

        # 可视化其中部分样本

        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = len(classes)
        samples_per_class = 7      # 每个类别可视化 7 个样本
        for y, clas in enumerate(classes):
            print "\n\ny =", y, "class =", clas
            idxs = np.flatnonzero(y_train == y)    # 转为1维数组后 y_train = y 的元素下标
            print "In training set, the number of class <", y, "> is", len(idxs)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)    # 从中随机选取 7 个样本
            print "randomly select", samples_per_class, "samples in this set, these subscript is:", idxs[:]
            print "plt_index = ",
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                print plt_idx,
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(X_train[idx].astype('uint8'))    # 显示图片
                plt.axis('off')
                if i == 0:
                    plt.title(clas)
        # 保存图片
        if not os.path.exists("visual_CIFAR10.jpg"):
            plt.savefig("visual_CIFAR10.jpg")
            plt.show()


        # 在本次实验中为了使得训练速度更快，因此抽样训练前5000个样本
        print "sampling...   reshape..."
        X_train = X_train[:5000]
        y_train = y_train[:5000]
        X_test = X_test[:500]
        y_test = y_test[:500]

        # 将矩阵 reshape 成 (nb_samples, nb_features) 的形式
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        print "X_train.shape =", X_train.shape, "X_test.shape =", X_test.shape

        # 赋值
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print "----"

    # ***
    def compute_distance(self, n_choose=3):
        """
            计算所有训练样本(nb_train)和所有测试样本(nb_test)之间的距离
            结果矩阵的维度为 nb_test*nb_train
        """
        print "\n----\n计算训练数据和测试数据中样本的距离..."
        nb_train = self.X_train.shape[0]
        nb_test = self.X_test.shape[0]
        dists = np.zeros((nb_test, nb_train))    # 距离矩阵

        # 计算距离方法1
        if n_choose == 1:
            for i in xrange(nb_test):
                for j in xrange(nb_train):
                    dists[i][j] = np.sqrt(np.sum(np.square(self.X_train[j,:] - self.X_test[i,:])))

        # 计算距离方法2  (重要)
        if n_choose == 2:
            for i in xrange(nb_test):
                dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - self.X_test[i, :]), axis=1))

        # 计算距离方法3  **(难) 凑出一个 a^2+b^2-2ab = (a-b)^2
        if n_choose == 3:
            dists = np.multiply(np.dot(self.X_test, self.X_train.T), -2)
            sq1 = np.sum(np.square(self.X_test), axis=1, keepdims=True)
            sq2 = np.sum(np.square(self.X_train), axis=1)   # 1*nb_train
            dists = np.add(dists, sq1)
            dists = np.add(dists, sq2)
            dists = np.sqrt(dists)

        self.dists = dists

        # We can visualize the distance matrix: each row is a single test example and
        # its distances to training examples
        #plt.imshow(dists)
        #plt.show()

        # To ensure that our vectorized implementation is correct, we make sure that it
        # agrees with the naive implementation. There are many ways to decide whether
        # two matrices are similar; one of the simplest is the Frobenius norm. In case
        # you haven't seen it before, the Frobenius norm of two matrices is the square
        # root of the squared sum of differences of all elements; in other words, reshape
        # the matrices into vectors and compute the Euclidean distance between them.
        #difference = np.linalg.norm(dists - dists_one, ord='fro')
        #print 'Difference was: %f' % (difference, )
        #if difference < 0.001:
        #    print 'Good! The distance matrices are the same'
        #else:
        #    print 'Uh-oh! The distance matrices are different'

        print "训练数据和测试数据的距离矩阵 dist.shape =", self.dists.shape   # (500, 5000)
        print "----"

    #
    def test_compute_distance_time(self):
        """
            测试 self.compute_distance 中三种方案的运行时间
        """
        start = time.time()
        self.compute_distance(n_choose=1)
        end = time.time()
        print "choose 1's time: ", end-start, "s"   # 73s

        start = time.time()
        self.compute_distance(n_choose=2)
        end = time.time()
        print "choose 2's time:  ", end-start, "s"  # 68s

        start = time.time()
        self.compute_distance(n_choose=3)
        end = time.time()
        print "choose 3's time:  ", end-start, "s"  # 0.55s

    #
    def knn_predict(self, k=3):
        """
            使用 CIFAR10 中的数据，用 KNN 分类器进行预测
        """
        self.nb_test = self.dists.shape[0]
        print "开始预测...  共计", self.nb_test, " 个样本..."
        y_pred = np.zeros(self.nb_test, dtype="uint8")
        for i in xrange(self.nb_test):
            # 与该测试样本距离最小的前 K 个训练样本的下标
            closest_idx = np.argsort(self.dists[i])[:k]
            # 这些下标对应的样本的 label
            closest_y = self.y_train[closest_idx]
            """
                出现次数最多的整数元素  np.bincount  np.argmax  np.unique
            """
            #y_pred[i] = np.argmax(np.bincount(closest_y))

            # 或者利用
            n, n_count = np.unique(closest_y, return_counts=True)
            y_pred[i] = n[np.argmax(n_count)]

        print "预测完毕..."
        return y_pred

    #
    def knn_accurary(self):
        """
            计算准确率
        """
        # k = 1
        y_test_pred = self.knn_predict(k=1)
        num_correct = np.sum(y_test_pred == self.y_test)    # 正确的样本个数
        print "计算准确率 k = 1: "
        accuracy = float(num_correct) / self.nb_test
        print 'Got %d / %d correct => accuracy: %f' % (num_correct, self.nb_test, accuracy)
        # about 0.274000

        # k = 5
        y_test_pred = self.knn_predict(k=5)
        num_correct = np.sum(y_test_pred == self.y_test)
        print "计算准确率 k = 5: "
        accuracy = float(num_correct) / self.nb_test
        print 'Got %d / %d correct => accuracy: %f' % (num_correct, self.nb_test, accuracy)
        print "----"
        # about 0.278000


#
if __name__ == '__main__':
    knn = KNN_classifier()
    # 1.0 载入数据， 可视化其中的部分图片
    knn.load_CIFAR10()
    # 2.0 计算训练数据和测试数据之间的距离
    knn.compute_distance()
    knn.test_compute_distance_time()    # 测试各种方法的运行时间

    # 3.0 预测测试集
    #knn.knn_predict()

    # 4.0 准确率
    knn.knn_accurary()


