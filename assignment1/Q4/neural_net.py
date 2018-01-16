from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
    这是个两层全连接神经网络，输入层有N维，隐藏层有H维，分类的问题包含C个类别。我们用softmax损失函数和权重矩阵的L2
    正则化来训练这个网络。网络在第一个完全连接层后使用ReLu作为激活函数。

    In other words, the network has the following architecture:
    换句话说，这个网络包含了以下结构：
    input - fully connected layer - ReLU - fully connected layer - softmax
    输入层 - 全连接层 -ReLu - 全连接层 - softmax
    The outputs of the second fully-connected layer are the scores for each class.
    两层神经网络的输出是每一个类别的分值。
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        初始化模型。权重初始化为很小的随机值，偏置项初始化为零。权重和偏置值都存储在self.params这个字典变量里，这个字典变量
        包含了以下的钥匙：

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        W1: 第一层的权重，大小为（D,H）
        b1: 第一层的偏置，大小为（H,)
        W2: 第二层的权重，大小为（H,C）
        b2: 第二层的偏置，大小为（C,）
        
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        输入：
        - input_size：输入数据的纬度为D
        - hidden_size：隐藏层的神经元数量H
        - output_size：分类数量C
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
        计算两层全连接神经网络的损失值和梯度

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.
        输入：
        - X：输入数据的纬度为（N，D），每一行是一个训练样本（也就是一副图像）
        - Y：类别标签向量。y[i]是X[i]的类别。每一个y[i]是一个在0和C之间的实数。这个值是可选的，如果不提供，
            我们就只是返回分值，如果提供，就返回损失值和梯度
        - reg：正则化强度

        Returns:
        返回：
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
        如果y是None，那就返回大小为（N,C）的分值矩阵，矩阵上[i，c]位置的值是绚练样本x[i]在第C类上的分值

        If y is not None, instead return a tuple of:
        如果y不是none，那就返回下面的值：
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - loss：这个小训练样本的数据损失和正则化损失
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        - grads：将参数名和这些参数的损失函数梯度对应起来存储的字典变量；它具有与self.params相同的钥匙。
        """
        # Unpack variables from the params dictionary
        # 从字典变量params中抓取变量值
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        # 计算向前传播的值
    
        scores = None
        
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # TODO：实现向前传播的值，计算输入数据每一类别的分值。将结果存储在分值变量中，分值变量是一个大小为
        #（N,C）的数组
        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1) 
        scores = a1.dot(W2) + b2
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    
        # If the targets are not given then jump out, we're done
        # 如果函数没有传入y的值，直接返回分值矩阵
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        #TODO：完成正向传播，并计算损失值。这应该包括数据损失以及W1和W2的L2正则化损失。将结果存储在损失变量中，
        #它是一个标量。使用Softmax分类器的损失函数
        # compute the class probabilities
        # 计算每个类别的概率值
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 

        # average cross-entropy loss and regularization
        # 交叉熵损失平均值和正则化
        corect_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(corect_logprobs) / N
        reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
        loss = data_loss + reg_loss
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        # 反向传播：计算梯度
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # TODO：计算反向传播，计算每个权重和偏置项的导数。将结果存在名为grads的字典。比如，grads['W1']
        # 应该存W1的梯度，矩阵大小与W相同
       
        # 计算分值的梯度
        dscores = probs
        dscores[range(N),y] -= 1
        dscores /= N

       
        # W2 和 b2
        grads['W2'] = np.dot(a1.T, dscores)
        grads['b2'] = np.sum(dscores, axis=0)
        
        # 反向传播里第二个隐藏层
        dhidden = np.dot(dscores, W2.T)
        
        # 激活函数ReLu的导数
        dhidden[a1 <= 0] = 0
        
        # 关于W1和b1的梯度
        grads['W1'] = np.dot(X.T, dhidden)
        grads['b1'] = np.sum(dhidden, axis=0)

        # 加上正则化梯度的部分
        grads['W2'] += reg * W2
        grads['W1'] += reg * W1
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
        使用随即梯度法训练神经网络
        Inputs:
        输入：
        - X: A numpy array of shape (N, D) giving training data.
        - X：大小为（N,D）的训练数据数组
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - y: 大小为（N,）的训练标签数组；y[i]=c意味着x[i]的标签为c，并且c在0和C之间
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - X_val: 大小为（N_val,D)的验证数据数组
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - y_val：大小为(N_val,)的验证标签数组
        - learning_rate: Scalar giving learning rate for optimization.
        - learning rate：在找最值过程中的学习率，是一个标量
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - learning_rate_decay：学习率衰减因子
        - reg: Scalar giving regularization strength.
        - reg：正则化强度，是个标量
        - num_iters: Number of steps to take when optimizing.
        - num_iters：寻找最值过程中的迭代次数
        - batch_size: Number of training examples to use per step.
        - batch_size：每一次迭代中用到的训练数据大小
        - verbose: boolean; if true print progress during optimization.
        - verbose：布尔变量；找最值过程中，如果正确就打印progress
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # TODO: 随机创建小批量的训练数据和训练标签，将它们各自存放在X_batch和y_batch里
            sample_indices = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[sample_indices]
            y_batch = y[sample_indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            # 使用目前的小批量数据计算损失值和梯度
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # TODO：使用grads字典中的梯度更新网络的参数（存储在self.params字典中）
            # 您将会用到上面计算出来的存在grads字典中的梯度。
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            # 每一次迭代，都计算训练和验证准确率，并且更新学习率
            if it % iterations_per_epoch == 0:
                # Check accuracy
                # 准确率计算
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                # 减小学习率
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
        使用训练得到的两层神经网络权重进行预测。对于每一个数据样本，我们预测它在每一个类别上的分值，
        并且将这个样本归于分值数据最高的类别。

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.
        输入：
        - X：一个大小为（N，D）的numpy数组，里面包含N维的需要分类的数据集

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        返回：
        - y_pred: 大小为（N，）的numpy数组，包含对每一副图像的预测类别。对于所有的i来说，y_pred[i]=c意味着
        X[i]的预测类别为c，当然c是在0和C之间。
        """
        
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # TODO：实现这个函数；这应该非常简单了！
        z1 = X.dot(self.params['W1']) + self.params['b1']
        a1 = np.maximum(0, z1) # pass through ReLU activation function
        scores = a1.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred
