import numpy as np
from random import shuffle

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
      
      softmax损失函数，简单的实现（包含迭代）
      
      输入图像的维度为D，一共有C个类别，图像个数为N
      
      输入：
      - W：大小为D x C的权重矩阵
      - X：大小为N x D的图像矩阵，一行表示一副图像
      - y：大小为N的一维数组，包含了从0到k-1的k个标签
      - reg: 正则化强度，是一个浮点数
      输出：
      一个元组：
      - 损失值，是一个浮点数
      - 相对于权重W的梯度，是一个与w同样大小的数组
    """
    # Initialize the loss and gradient to zero.
    # 先将损失值和梯度初始化为0
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #使用显式循环计算softmax损失及其梯度。损失值存在变量loss里，梯度存在dW里。如果你不小心的话，
    #很可能会遇到数值不稳定的问题。不要忘记正则化！
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    loss = 0.0
    # 对于图像向量中的每一幅图像都进行损失值计算，再将每一幅图像的损失值加总，所以这是一个迭代的过程
    for i in xrange(num_train):
        # 计算分值向量
        f_i = X[i].dot(W)
        # 为了避免数值不稳定性问题，将向量里的每一个值都减去向量中的最大值，参考知识点4
        f_i -= np.max(f_i)

        # 计算损失值
        sum_j = np.sum(np.exp(f_i))
        p = lambda k: np.exp(f_i[k]) / sum_j #lambda函数是一类特殊函数
        loss += -np.log(p(y[i]))

        # 计算梯度
        for k in range(num_classes):
            p_k = p(k)
            dW[:, k] += (p_k - (k == y[i])) * X[i]
    
    # 损失值等于平均后的损失值加上正则项
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #使用非显式循环计算softmax损失值及其梯度。损失值存在变量loss里，梯度存在dW里。如果你不小心的话，
    #很可能会遇到数值不稳定的问题。不要忘记正则化！
    num_train = X.shape[0]
    f = X.dot(W)
    f -= np.max(f, axis=1, keepdims=True) # 如上
    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    p = np.exp(f)/sum_f

    loss = np.sum(-np.log(p[np.arange(num_train), y]))

    ind = np.zeros_like(p)
    ind[np.arange(num_train), y] = 1
    dW = X.T.dot(p - ind)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW