<font size=3>在使用卷积神经网络进行分类任务时，往往使用以下几类损失函数：

 - **平方误差损失**
 - **SVM损失**
 - **softmax损失**

<font size=3>其中，平方误差损失在分类问题中效果不佳，一般用于回归问题。softmax损失函数和SVM(多分类)损失函数在实际应用中非常广泛。本文将对这两种损失函数做简单介绍，包括<font color=red>损失函数的计算、梯度的求解以及python中使用Numpy库函数进行实现。</font>


#SVM多分类
##<font color=blue>1. 损失函数</font>
<font size=3>一般而言，深度学习中使用的SVM损失函数是基于 [Weston and Watkins 1999 (pdf)](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es1999-461.pdf) 。
<font size=3>其损失函数如下：

<font size=5>$$L_i = \sum_{j\neq y_i}max(f_j - f_{y_i} + \Delta)$$</font>

在实际使用中，$\Delta$ 的值一般取1，代表间隔。

<font size=3>在神经网络中，由于我们的评分函数是: </font>
<font size=5>$$f = W * x$$ </font>
<font size=3>因此，可以将损失函数改写如下:</font>
<font size=5> $$L_i = \sum_{j\neq y_i}max(W_j^Tx_i+W_{y_i}^Tx_i + \Delta)$$
</font>
<font size=3>如果考虑整个训练集合上的平均损失，包括正则项，则公式如下：</font>
<font size=5>$L=\frac{1}{N}\sum_i\sum_{j\not=y_i}[max(0,f(x_i;W)_j-f(x_i;W)_{y_i}+\Delta)]+\lambda \sum_k \sum_l W^2_{k,l}$</font>


><font color=red>直观理解:</font>
><font size=3>多类SVM“想要”正确类别的分类分数比其他不正确分类类别的分数要高，而且至少高出delta的边界值。如果其他分类分数进入了红色的区域，甚至更高，那么就开始计算损失。如果没有这些情况，损失值为0。我们的目标是找到一些权重，它们既能够让训练集中的数据样例满足这些限制，也能让总的损失值尽可能地低。</font>
![这里写图片描述](http://img.blog.csdn.net/20161226224026495?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYmNqMjk2MDUwMjQw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

<br><br>
**<font color=green size=4>举一个具体的例子：</font>**
<br><font size=3>例子来源于 **斯坦福CS231n** 课件。第一张图片是猫，神经网络计算得出其三个类别的分值分别为 3.2, 5.1 和 -1.7。很明显，理想情况下猫的分值应该高与其他两种类别，但根据计算结果，car的分值最高，因此在当前的权值设置下，该 network 会把这张图片分类为 car。此时我们可以根据公式计算损失
<br><font color=red size=3>损失计算如下：(S代表Score，即分值)</font>
<font size=5>$$L_i = max(0, S_{car} - S_{cat}+\Delta ) + max(0, S_{frog} - S_{cat} + \Delta) = 2.9 + 0$$</font>

![这里写图片描述](http://img.blog.csdn.net/20161226222452176?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYmNqMjk2MDUwMjQw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

<br>
###<font color=blue>2. 梯度公式推导</font>
<font color=red size=3.5>设置以下变量：</font>
- <font size=3>矩阵 $W$ 代表权值，维度是 $D*C$，其中 $D$ 代表特征的维度，$C$ 代表类别数目。
- <font size=3>矩阵 $X$ 代表样本集合，维度是 $N*D$， 其中 $N$ 代表样本个数。
- <font size=3>分值计算公式为 $f = X*W$，其维度为 $N*C$, 每行代表一个样本的不同类别的分值。

<font size=4>对于第 $i$ 个样本的损失函数计算如下:</font>
<font size=5>$$L_i =\sum_{j\neq y_i}max(0, W_{:,j}^Tx_{i, :} - W_{:,y_i}^Tx_{i, :}+\Delta)$$</font>

<font color=red size=4>偏导数计算如下:</font><br>

<font size=5>$$\frac{\partial L_i}{\partial W_{:,y_i}} = -(\sum_{j\not=y_i}1(w^T_{:,j}x_{i,:}-w^T_{:,y_i}x_{i,:}+\Delta>0))x_{i,:}$$</font>
<br>
<font size=5>$$\frac{\partial L_i}{\partial W_{:,j}} = 1(w^T_{:,j}x_{i,:}-w^T_{:,y_i}x_{i,:}+\Delta>0)x_{i,:}$$</font>

<font size=3.5 color=green>其中：
- $ w_{:,j}$ 代表W矩阵第 $j$ 列，其维度为 $D$。
- $ x_{i,:}$ 代表X矩阵的第 $i$ 行，表示样本 $i$ 的特征，其维度也为 $D$ 。
二者相乘，得出的是样本 $i$ 在第 $j$ 个类别上的得分。
- $1$ 代表示性函数。
</font>

###<font color=blue>3. python实现</font>
包括向量化版本和非向量化版本：
```python

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
```

<br><br>
#Softmax 损失函数
##<font color=blue>1. 损失函数</font>
<font size=3>
Softmax 函数是 Logistic 函数的推广，用于多分类。

<font size=3>分值的计算公式不变：</font>
<font size=5>$$f(xi; W) = W*x$$</font>

<font size=3>损失函数使用交叉熵损失函数，第 $i$ 个样本的损失如下：</font><br>
<font color=red size=5>$$L_i = -log(\frac{e^{f_{y_i}}}{\sum_j e^{f_j}})$$</font>

<font size=3>其中正确类别得分的概率可以被表示成：</font><br>
<font color=red size=5> $$P(y_i | x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j}}$$ </font>

><font size=3>在实际使用中，$e^{f_j}$ 常常因为指数太大而出现<font color=red>数值爆炸</font>问题，两个非常大的数相除会出现<font color=red>数值不稳定</font>问题，因此我们需要在分子和分母中同时进行以下处理：
><font size=5>$$\frac{e^{f_{y_i}}}{\sum_j e^{f_j}} = \frac{Ce^{f_{y_i}}}{C\sum_j e^{f_j}} = \frac{e^{f_{y_i}+logC}}{\sum_j e^{f_j+logC}}$$ </font>
>其中$C$ 的设置是任意的，在实际变成中，往往把$C$设置成：
><font size=5>$$logC = -max f_j$$</font>
即第 $i$ 个样本所有分值中最大的值，当现有分值减去该最大分值后结果$\leq0$，放在 $e$ 的指数上可以保证分子分布都在<font color=red>0-1</font>之内。

##<font color=blue>2. 梯度推导</font>
梯度的推导如下：
![这里写图片描述](http://img.blog.csdn.net/20161227105554766?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYmNqMjk2MDUwMjQw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

##<font color=blue>3. Python实现</font>

```
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

    loss = 0.0
    dW = np.zeros_like(W)
    dW_each = np.zeros_like(W)
    #
    num_train, dim = X.shape
    num_class = W.shape[1]
    f = X.dot(W)        # 样本数*类别数   分值
    #
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))
    # 计算对数概率  prob.shape=N*10  每一行与一个样本相对应  每一行的概率和为1
    # 其中 f_max 是每行的最大值，exp(x)中x的值过大而出现数值不稳定问题
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
    #
    y_trueClass = np.zeros_like(prob)
    y_trueClass[np.arange(num_train), y] = 1.0     # 每行只有正确的类别处为1，其余为0
    #
    for i in range(num_train):
        for j in range(num_class):
            loss += -(y_trueClass[i, j] * np.log(prob[i, j]))
            dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i, :]
        dW += dW_each
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    loss = 0.0
    dW = np.zeros_like(W)    # D by C
    num_train, dim = X.shape

    f = X.dot(W)    # N by C
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # N by 1
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
    y_trueClass = np.zeros_like(prob)
    y_trueClass[range(num_train), y] = 1.0    # N by C

    # 计算损失  y_trueClass是N*C维度  np.log(prob)也是N*C的维度
    loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)

    # 计算损失  X.T = (D*N)  y_truclass-prob = (N*C)
    dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W

    return loss, dW
```


<br><br>
##Softmax、SVM损失函数用于CIFAR-10图像分类
<font color=blue size=3.5>CIFAR-10 小图分类是对于练习而言非常方便的一个数据集。通过在该数据集上实现基本的 softmax 损失函数 和 SVM 损失函数以及可视化部分结果，可以加深对算法的理解。

<font color=red size=4> 关于本文的全部代码可以到[GitHub中下载](https://github.com/Baichenjia/CIFAR-10-basic)</font>

<font size=3>下面给出代码运行过程中的输出结果：

###<font color=red>1. 可视化CIFAR-10的部分样本
![这里写图片描述](http://img.blog.csdn.net/20161227113250764?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYmNqMjk2MDUwMjQw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

###<font color=red>原始像素作为特征使用SVM分类的损失图
![这里写图片描述](http://img.blog.csdn.net/20161227113356354?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYmNqMjk2MDUwMjQw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

###<font color=red>两层神经网络使用softmax分类的损失和准确率图
![这里写图片描述](http://img.blog.csdn.net/20161227113450980?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYmNqMjk2MDUwMjQw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

###<font color=red>两层神经网络使用softmax分类的第一个隐含层权重图：
![这里写图片描述](http://img.blog.csdn.net/20161227113743374?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYmNqMjk2MDUwMjQw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#参考资料
>[1] http://www.jianshu.com/p/004c99623104
>[2] http://deeplearning.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92
>[3] http://blog.csdn.net/acdreamers/article/details/44663305
>[4] http://cs231n.github.io/

#结束
欢迎邮件交流 bai_chenjia@163.com
