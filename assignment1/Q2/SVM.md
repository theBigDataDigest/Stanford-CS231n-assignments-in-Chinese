# SVM作业记录

标签（空格分隔）： 未分类


> 之前的同学，已经将运行官方给定程序的注意事项说的很明白了，还有疑问呢，在svm.ipynb里面也有注释，这个文档主要对 linear SVM里面注意的事项，结合课程的笔记，做一些说明。


## 任务
# Multiclass Support Vector Machine exercise

*Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. For more details see the [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html) on the course website.*

当你提交作业时，完成并提交此工作表（包括它的输出以及任何用到的在此工作表外的code）。更多的细节请看课程网页：
[assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html) 



In this exercise you will:
    
- implement a fully-vectorized **loss function** for the SVM
- implement the fully-vectorized expression for its **analytic gradient**
- **check your implementation** using numerical gradient
- use a validation set to **tune the learning rate and regularization** strength
- **optimize** the loss function with **SGD**
- **visualize** the final learned weights

在这次练习中，你将会学到：  

- 完成一个基于SVM的全向量化损失函数  
- 完成解析梯度的全向量化表示  
- 用数值化的梯度来检验你的完成情况  
- 使用一个验证集去调优学习率和正则化强度
- 运用随机梯度下降去优化损失函数
- 可视化最后的学习得到的权重  

## 知识点
### 1. 线性SVM分类器  
此处讲的SVM视角，与一般机器学习不同，它是以Score的视角来看待彼此类别的差异。在官方的note中也提供了多分类SVM实现的多个版本的内容，这个版本的SVM是来自于 [Weston and Watkins 1999 (pdf)](http://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es1999-461.pdf),据说比OVA版本的SVM好，具体见官方notes。这并不是这次课的主要内容，强调的是：使用线性评分函数得分的相对大小来衡量样本点属于各个类别的可能性的差异。至少斯坦福CS231N课程对这个结论进行了很给力的背书。  

可以简单认为，正确的分类分数，应该比错误的分类分数大，有一定的阈值，这个阈值是`$\Delta$`，虽然错误分类的分数比正确分类的分数大，但不大于`$\Delta$`，那么损失为0。自然地，如果错误分类的分数比正常分类损失小，损失为0，当错误分类的得分情况不属于这2种情况是，产生SVM损失，hinge损失。  

hinge损失公式为

```math
L_j = \sum_{j\neq y_i}max(0,s_j-s_{y_i}+\Delta )

```


而当我们中心化线性方程`$f(x)=w^Tx+b$` 为 `$f(x)=w^Tx$`后，可以发现，随着改变`$w$`的大小，线性方程的形式展示上的数值大小只是某个`$w$`的倍数形式。而得分函数`$s$`的得分有意义的是相对值，而不是绝对值，只要`$w$`乘于任何比例系数，得分将**整体**变化，分类之间的差值也将变化。多分类的大小关系，只有**大于、小于、等于** 三种，所以，相对得分关系`$(s_j-s_{y_i}+\Delta)$`,里面的`$\Delta$`可以直接设定为1(简单回忆一下高中数学里面的“不妨设为” :D。  而且最后考虑到正则化因子的时候，真正控制权重的范围的，应该是正则化超参数`$\lambda$` 。  

**总结一下：**
  
1.   使用线性评分函数的得分大小进行分类； 
2.   如果想将对于所有类别都分类正确即对于所有`$s_{y_i}-s_j>=1$`,再引入hinges损失；  
3.   然后再加入一个正则项来衡量这个模型的复杂度，学过线性回归的同学自然想到正则化的L1或者L2正则，比如`$\frac{1}{2}\|w\|$`，然后目标函数就变成：  


```math
\sum_{j\neq y_i}max(0,s_j-s_{y_i}+\Delta ) + \frac{\lambda}{2}\|w\|
```

  

到了这一步，基本上能看出一些与传统机器学习推导SVM公式的思路异曲同工的地方了，至于求解，用SGD就可以了。   



以上，就是本次多分类SVM的一些心得，要细究起来，水还是很深的。


#### 接上一位同学的前置步骤，即展示出分类图片样子，下面开始进行继续骚操作   
- Split the data into train, val, and test sets. In addition we will
- create a small development set as a subset of the training data;
- we can use this for development so our code runs faster.


翻译：将数据分割为训练，验证和测试集。另外我们将创建一个小的“发展集”作为训练集的子集，这样我们能用发展集使得我们的代码运行更快。  

老实说：吴恩达说过这个development set 也可以叫做diff on most called validation set 应该就是validation set的一种形式，不知道如何翻译，暂叫发展集吧（只限于这一章），因为最近看吴恩达老师的视频他黑板上写得不是val，而是train/dev/test, 所以，在正常的机器学习流程中应该将dev理解成验证集。  


个人看法，这个developmen set在这个note book里 感觉就是从训练集中进行了一个小的抽样，用来测试一些结论

```python


num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Our validation set will be num_validation points from the original
# training set.验证集将会是从原始的训练集中分割出来的可数的数据样本点
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.训练集用的是开始的数据可训练的样本点、
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.我们也可以从训练集中随机抽取一小部分的数据点作为发展集
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# We use the first num_test points of the original test set as our
# test set. 使用前num_test个训练集点作为训练集
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
```

    Train data shape:  (49000, 32, 32, 3)
    Train labels shape:  (49000,)
    Validation data shape:  (1000, 32, 32, 3)
    Validation labels shape:  (1000,)
    Test data shape:  (1000, 32, 32, 3)
    Test labels shape:  (1000,)
  




    
```python
# Preprocessing: reshape the image data into rows，数据预处理，将每副图像数据重塑性为一行数据。
#np.reshape(kkk,-1),其中kkk为任意有具体含义的数字时，-1表示并不因为指定，由前一个参数和总的参数决定“-1”位置代表的数据。
#将所有样本，各自拉成一个行向量，所构成的二维矩阵，每一行就是一个样本，即一行有32X32X3个列，每一列表示一个特征。
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
#很正常地，要打印出来数据的形状看看。
print('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)
```

    Training data shape:  (49000, 3072)
    Validation data shape:  (1000, 3072)
    Test data shape:  (1000, 3072)
    dev data shape:  (500, 3072)



```python
# Preprocessing: subtract the mean image，预处理，减去图像的平均值，中心化有两种，一种是3个通道颜色的中心值各是多少，一种是图片上每个特征的平均值是多少。
# first: compute the image mean based on the training data，首先，基于训练数据，计算图像的平均值
mean_image = np.mean(X_train, axis=0)#计算每一列特征的平均值，共32x32x3个特征
print(mean_image.shape)
print(mean_image[:10]) # print a few of the elements，查看一下特征的数据
plt.figure(figsize=(4,4))#指定画图的框图大小
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image，将平均值可视化出来。
plt.show()
```

    (3072,)
    [ 130.64189796  135.98173469  132.47391837  130.05569388  135.34804082
      131.75402041  130.96055102  136.14328571  132.47636735  131.48467347]
        
        
        
        
```python
# second: subtract the mean image from train and test data#
#去中心化
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image
```


```python
# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
#把偏置b放进x里面去，这样就能统一方程形式。b当成X的一部分，x取值恒为1.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)#最后得到各个数据集的shape
```

    (49000, 3073) (1000, 3073) (1000, 3073) (500, 3073)


## SVM Classifier

Your code for this section will all be written inside **cs231n/classifiers/linear_svm.py**. 

As you can see, we have prefilled the function `compute_loss_naive` which uses for loops to evaluate the multiclass SVM loss function. 

开始写code环节了，这部分的code在 xxx/cs231n/classifiers/linear_svm.py 里，请按要求补充


```python
# Evaluate the naive implementation of the loss we provided for you:
#计算我们提供给你的简单的损失
#这里如果有一个报错，2.7因为你要引入一个past的库，但是这个呢。。。在future里。。。所以你要pip install future，所以要多用google 
#python3中 xrange已经改为range。。。
from cs231n.classifiers.linear_svm import svm_loss_naive
import time

# generate a random SVM weight matrix of small numbers
#真的很小啊，先标准正态随机然后取0.0001,这种初始化有问题，10层顶天了，具体见后面课程
W = np.random.randn(3073, 10) * 0.0001 

loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)#从dev数据集种的样本抽样计算的loss是。。。大概估计下多少，随机几次，loss在8至9之间
print('loss: %f' % (loss, ))
```

    loss: 9.417096


The `grad` returned from the function above is right now all zero. Derive and implement the gradient for the SVM cost function and implement it inline inside the function `svm_loss_naive`. You will find it helpful to interleave your new code inside the existing function.

翻译：从上面返回的`grad`现在全是0，完成SVM损失函数并取得相应的值，函数在`svm_loss_native`.将你的新code插入已有的完成完成任务。

To check that you have correctly implemented the gradient correctly, you can numerically estimate the gradient of the loss function and compare the numeric estimate to the gradient that you computed. We have provided code that does this for you:

翻译：为了检验你是否已经正确地完成了梯度算法，你可以用数值方法估算损失方程的梯度然后比较数值与你用方程计算的值，我们已经将code 在下方提供给你。

```python
# Once you've implemented the gradient, recompute it with the code below
# and gradient check it with the function we provided for you
# Compute the loss and its gradient at W.

#输出是grad_check_sparse函数的结果,2种情况下，可以看出，其实2种算法误差已经几乎不计了。。。
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)
# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
from cs231n.gradient_check import grad_check_sparse

f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]

grad_numerical = grad_check_sparse(f, W, grad)

# do the gradient check once again with regularization turned on
# you didn't forget the regularization gradient did you?
loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)

f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]

grad_numerical = grad_check_sparse(f, W, grad)
```

    numerical: 4.374474 analytic: 4.374474, relative error: 1.015691e-12
    numerical: 6.078824 analytic: 6.078824, relative error: 5.609252e-11
    numerical: 5.147538 analytic: 5.147538, relative error: 9.815960e-12
    numerical: -9.483265 analytic: -9.483265, relative error: 7.616785e-12
    numerical: -22.481870 analytic: -22.481870, relative error: 7.357678e-12
    numerical: -5.599093 analytic: -5.599093, relative error: 5.088738e-12
    numerical: 14.433376 analytic: 14.433376, relative error: 1.940539e-11
    numerical: -16.998000 analytic: -16.998000, relative error: 2.118555e-11
    numerical: -28.237744 analytic: -28.237744, relative error: 1.651817e-12
    numerical: 35.408621 analytic: 35.408621, relative error: 8.001593e-12
    
    numerical: 27.742121 analytic: 27.738151, relative error: 7.155358e-05
    numerical: -40.339974 analytic: -40.338987, relative error: 1.223576e-05
    numerical: 0.845399 analytic: 0.849856, relative error: 2.629202e-03
    numerical: 27.420879 analytic: 27.419438, relative error: 2.627064e-05
    numerical: -18.446754 analytic: -18.446542, relative error: 5.762364e-06
    numerical: -35.832186 analytic: -35.831168, relative error: 1.421181e-05
    numerical: 0.832914 analytic: 0.834876, relative error: 1.175947e-03
    numerical: -43.613155 analytic: -43.623197, relative error: 1.151126e-04
    numerical: -26.350742 analytic: -26.344371, relative error: 1.209057e-04
    numerical: 22.810233 analytic: 22.804276, relative error: 1.305981e-04


### Inline Question 1:
It is possible that once in a while a dimension in the gradcheck will not match exactly. What could such a discrepancy be caused by? Is it a reason for concern? What is a simple example in one dimension where a gradient check could fail? *Hint: the SVM loss function is not strictly speaking differentiable*

**Your Answer:** *解析解和数值逼近解的区别，而数值解是用前后2个很小的随机尺度(比如0.00001)进行计算，当Loss不可导的，两者会出现差异。*


```python
# Next implement the function svm_loss_vectorized; for now only compute the loss;
# we will implement the gradient in a moment.
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

from cs231n.classifiers.linear_svm import svm_loss_vectorized
tic = time.time()
loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# The losses should match but your vectorized implementation should be much faster.
print('difference: %f' % (loss_naive - loss_vectorized))
```

    Naive loss: 9.417096e+00 computed in 0.121770s
    Vectorized loss: 9.417096e+00 computed in 0.005394s
    difference: 0.000000



```python
# Complete the implementation of svm_loss_vectorized, and compute the gradient
# of the loss function in a vectorized way.

# The naive implementation and the vectorized implementation should match, but
# the vectorized version should still be much faster.
#结论：结果一样，向量化快一些
tic = time.time()
_, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss and gradient: computed in %fs' % (toc - tic))

tic = time.time()
_, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss and gradient: computed in %fs' % (toc - tic))

# The loss is a single number, so it is easy to compare the values computed
# by the two implementations. The gradient on the other hand is a matrix, so
# we use the Frobenius norm to compare them.
difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('difference: %f' % difference)
```

    Naive loss and gradient: computed in 0.127225s
    Vectorized loss and gradient: computed in 0.003415s
    difference: 0.000000


### Stochastic Gradient Descent

We now have vectorized and efficient expressions for the loss, the gradient and our gradient matches the numerical gradient. We are therefore ready to do SGD to minimize the loss.

我们已经向量化并且有效地表达了损失、梯度，而且我们的梯度是与梯度数值解相一致的。因此我们可以利用SGD来最小化损失了。

```python
# In the file linear_classifier.py, implement SGD in the function
# LinearClassifier.train() and then run it with the code below.
# 在文件 linear_classifier.py 里，完成SGD函数
# LinearClassifier.train() 然后用下面的code运行它。

from cs231n.classifiers import LinearSVM
svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print('That took %fs' % (toc - tic))
```

    iteration 0 / 1500: loss 406.551290
    iteration 100 / 1500: loss 240.613943
    iteration 200 / 1500: loss 146.111274
    iteration 300 / 1500: loss 90.213982
    iteration 400 / 1500: loss 56.328602
    iteration 500 / 1500: loss 35.504821
    iteration 600 / 1500: loss 23.727488
    iteration 700 / 1500: loss 15.755706
    iteration 800 / 1500: loss 11.425011
    iteration 900 / 1500: loss 9.055450
    iteration 1000 / 1500: loss 7.637563
    iteration 1100 / 1500: loss 6.671778
    iteration 1200 / 1500: loss 5.914076
    iteration 1300 / 1500: loss 5.219254
    iteration 1400 / 1500: loss 5.034812
    That took 3.781817s



```python
# A useful debugging strategy is to plot the loss as a function of
# iteration number:
# 大佬说了，有效的debug策略就是去将损失和循环次数画出来。
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()
```


![png](output_21_0.png)



```python
# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
# 编写函数 LinearSVM.predict，评估训练集和测试集的表现。

y_train_pred = svm.predict(X_train)
print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
y_val_pred = svm.predict(X_val)
print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))
```

    training accuracy: 0.380776
    validation accuracy: 0.392000



```python
# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.4 on the validation set.
# 使用验证集去调参（正则化强度和学习率），你将会试验各种不同的学习率
# 和正则化强度，如果你很谨慎，你将会在验证集上得到一个分类准确度大约是0.4的结果。
# 设置学习率和正则化强度，多设几个靠谱的，可能会好一点。

learning_rates = [1e-7, 3e-7, 5e-7, 7e-7, 9e-7, 5e-5]
regularization_strengths = [2e4, 3e4, 4e4, 5e4, 6e4]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.

results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#写下你的code ,通过验证集选择最佳超参数。对于每一个超参数的组合，在训练集训练一个线性svm，计 #
#算它的准确度在训练集和测试集，然后存储这些数字在结果字典里。另外，存储最好的验证集准确度在   #
#best_val和线性SVM模型object在best_svm.
#
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #

#提示：当你编写你的验证code时，你应该使用一个小的数据集作一些数字迭代,这样的话，SVM的训练模型#
#们并不会花费太多的时间去训练。一旦你确认你的验证code是可以正常工作的，你应该返回这些验证code，#
#并作用在大数据级别的迭代训练上。
################################################################################
for rate in learning_rates:
    for regular in regularization_strengths:
        svm = LinearSVM()
        svm.train(X_train, y_train, learning_rate=rate, reg=regular,
                      num_iters=1000)
        y_train_pred = svm.predict(X_train)
        accuracy_train = np.mean(y_train == y_train_pred)
        y_val_pred = svm.predict(X_val)
        accuracy_val = np.mean(y_val == y_val_pred)
        results[(rate, regular)]=(accuracy_train, accuracy_val)
        if (best_val < accuracy_val):
            best_val = accuracy_val
            best_svm = svm
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
    
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print ('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))
    
print ('best validation accuracy achieved during cross-validation: %f' % best_val)
```

    lr 1.000000e-07 reg 2.000000e+04 train accuracy: 0.371776 val accuracy: 0.381000
    lr 1.000000e-07 reg 3.000000e+04 train accuracy: 0.377347 val accuracy: 0.390000
    lr 1.000000e-07 reg 4.000000e+04 train accuracy: 0.374612 val accuracy: 0.383000
    lr 1.000000e-07 reg 5.000000e+04 train accuracy: 0.368122 val accuracy: 0.379000
    lr 1.000000e-07 reg 6.000000e+04 train accuracy: 0.363980 val accuracy: 0.380000
    lr 3.000000e-07 reg 2.000000e+04 train accuracy: 0.372122 val accuracy: 0.376000
    lr 3.000000e-07 reg 3.000000e+04 train accuracy: 0.366388 val accuracy: 0.369000
    lr 3.000000e-07 reg 4.000000e+04 train accuracy: 0.345633 val accuracy: 0.344000
    lr 3.000000e-07 reg 5.000000e+04 train accuracy: 0.356245 val accuracy: 0.366000
    lr 3.000000e-07 reg 6.000000e+04 train accuracy: 0.352020 val accuracy: 0.376000
    lr 5.000000e-07 reg 2.000000e+04 train accuracy: 0.346735 val accuracy: 0.365000
    lr 5.000000e-07 reg 3.000000e+04 train accuracy: 0.352939 val accuracy: 0.376000
    lr 5.000000e-07 reg 4.000000e+04 train accuracy: 0.340918 val accuracy: 0.329000
    lr 5.000000e-07 reg 5.000000e+04 train accuracy: 0.337898 val accuracy: 0.342000
    lr 5.000000e-07 reg 6.000000e+04 train accuracy: 0.337959 val accuracy: 0.338000
    lr 7.000000e-07 reg 2.000000e+04 train accuracy: 0.350980 val accuracy: 0.351000
    lr 7.000000e-07 reg 3.000000e+04 train accuracy: 0.299837 val accuracy: 0.317000
    lr 7.000000e-07 reg 4.000000e+04 train accuracy: 0.343306 val accuracy: 0.353000
    lr 7.000000e-07 reg 5.000000e+04 train accuracy: 0.286102 val accuracy: 0.268000
    lr 7.000000e-07 reg 6.000000e+04 train accuracy: 0.312673 val accuracy: 0.320000
    lr 9.000000e-07 reg 2.000000e+04 train accuracy: 0.307327 val accuracy: 0.327000
    lr 9.000000e-07 reg 3.000000e+04 train accuracy: 0.322816 val accuracy: 0.338000
    lr 9.000000e-07 reg 4.000000e+04 train accuracy: 0.282939 val accuracy: 0.287000
    lr 9.000000e-07 reg 5.000000e+04 train accuracy: 0.313776 val accuracy: 0.344000
    lr 9.000000e-07 reg 6.000000e+04 train accuracy: 0.300224 val accuracy: 0.307000
    lr 5.000000e-05 reg 2.000000e+04 train accuracy: 0.180347 val accuracy: 0.174000
    lr 5.000000e-05 reg 3.000000e+04 train accuracy: 0.105837 val accuracy: 0.113000
    lr 5.000000e-05 reg 4.000000e+04 train accuracy: 0.140612 val accuracy: 0.148000
    lr 5.000000e-05 reg 5.000000e+04 train accuracy: 0.064612 val accuracy: 0.073000
    lr 5.000000e-05 reg 6.000000e+04 train accuracy: 0.068980 val accuracy: 0.091000
    best validation accuracy achieved during cross-validation: 0.390000



```python
# Visualize the cross-validation results
#可视化交叉验证结果
import math
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
#画出训练准确度
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
#画出验证准确度
colors = [results[x][1] for x in results] # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()
```


![png](output_24_0.png)



```python
# Evaluate the best svm on test set
#评估best_svmz在测试集上的表现。
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)
```

    linear SVM on raw pixels final test set accuracy: 0.377000


###### 结果偏低，不过考虑到是一个十分类问题，均匀分布下，乱猜的结果是`$1/10$`,所以还是有那么一点意思的。当然此处并没有运用特征工程进行特征处理，也没有运用神经网络进行学习，这也是后面要干的事！。 ######

```python
# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
#对于每一类，可视化学习到的权重
#依赖于你对学习权重和正则化强度的选择，这些可视化效果或者很明显或者不明显。
w = best_svm.W[:-1,:] # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
      
    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
```


![png](output_26_0.png)


### Inline question 2:
Describe what your visualized SVM weights look like, and offer a brief explanation for why they look they way that they do.

**Your answer:** *将学习到的权重可视化的效果可以看到，权重是对于原图像的特征提取，与原图像关系很大，很朴素的思想，在分类器权重向量上投影最大的向量得分应该最高，训练样本训练出来的权重向量最好的结果就是训练点上提取出来的共性的方向。也就是一种模板，毕竟这在CNN里面有一个叫featuremap的东西。*






