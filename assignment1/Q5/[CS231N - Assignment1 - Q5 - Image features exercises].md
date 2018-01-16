# [CS231N - Assignment1 - Q5 - Image features exercises]



---

我们已经看到，通过用输入图像的像素训练的线性分类器对图像的分类问题已经取得了不错的结果。在这个练习中我们会用对图像像素进一步计算得来的特征来训练线性分类器从而提高性能。

## 抽取特征（Extract Features）

对于每张图，我们会计算**梯度方向直方图(HOG)**特征和用**HSV（Hue色调，Saturation饱和度,Value明度）**颜色空间的**色调**特征。把每张图的梯度方向直方图和颜色直方图特征合并形成我们最后的特征向量。


粗略的讲呢，HOG应该可以捕捉到图像的纹理特征而忽略了颜色信息，颜色直方图会表示图像的颜色特征而忽略了纹理特征(详细见[这篇](http://www.jianshu.com/p/395f0582c5f7))。所以我们预估把两者结合起来得到的效果应该是比用其中一种得到的效果好。对于后面的bonus，验证一下这个设想是不错的选择。


 ```hog_feature```和```color_histogram_hsv```两个函数都只对一张图做操作并返回这张图片的特征向量。```extract_features```函数接收一堆图片和一个list的特征函数，然后用每个特征函数在每张图片上过一遍，把结果存到一个矩阵里面，矩阵的每一**行**都是一张图片的所有特征的合并。【注：题目中写的column,但从实际结果上来看应该是行】

### 1. 代码解析
```python
from cs231n.features import *

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

print(X_train_feats.shape, X_val_feats.shape, X_test_feats.shape)

# 预处理：减掉每一列特征的平均值
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat


# 预处理：每一列除以标准差，这确保了每个特征都在一个数值范围内
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat


# 多加一个bias列
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])
```
在features.py中写了两个特征的计算方法，HOG是改写了scikit-image的fog接口，并且首先要转换成灰度图。颜色直方图是实现用matplotlib.colors.rgb_to_hsv的接口把图片从RGB变成HSV，再提取明度(value)，把value投射到不同的bin当中去。关于HOG的原理请谷歌百度。

### 可能会踩的坑
1. *"Import Error: cannot import name imread"* 安装或者重装Pillow或PIL(pip intall PIL)。如果安装提示已经安装了，可以重装，下载地址在[这里](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy), 下载后，pip install xxxx.whl
2. ``` orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx/2::cx, cy/2::cy].T```这行报错,*"TypeError: slice indices must be integers or None or have an \__index__ method"*,可以把代码改成: ```orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[int(cx/2)::cx, int(cy/2)::cy].T```

## SVM训练(Train SVM on Features)
 *Using the multiclass SVM code developed earlier in the assignment, train SVMs on top of the features extracted above; this should achieve better results than training SVMs directly on top of raw pixels.*
 用前面作业中的多分类SVM来对上面抽取到的特征进行训练。这次应该会比之前直接在像素上训练的结果更好。
 
```python
from cs231n.classifiers.linear_classifier import LinearSVM

#learning_rates = [1e-9, 1e-8, 1e-7]
#regularization_strengths = [5e4, 5e5, 5e6]

results = {}
best_val = -1
best_svm = None

#pass
learning_rates =[5e-9, 7.5e-9, 1e-8]
regularization_strengths = [(5+i)*1e6 for i in range(-3,4)]
################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained classifer in best_svm. You might also want to play          #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################
################################################################################
# TODO:                                                                        #
# 用验证集来调整learning rate和regularization的强度。                            #
# 这个应该和你之前做SVM做验证是一样的，把最好的结果保存在best_svm。你也许想试        #
# 试不同的颜色直方图的bin的个数。如果你调的够仔细，应该可以在验证集上得到            #
# 差不多0.44的正确率。                                                          #
################################################################################
for rs in regularization_strengths:
    for lr in learning_rates:
        svm = LinearSVM()
        loss_hist = svm.train(X_train_feats, y_train, lr, rs, num_iters=6000)
        y_train_pred = svm.predict(X_train_feats)
        train_accuracy = np.mean(y_train == y_train_pred)
        y_val_pred = svm.predict(X_val_feats)
        val_accuracy = np.mean(y_val == y_val_pred)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm           
        results[(lr,rs)] = train_accuracy, val_accuracy
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
print('best validation accuracy achieved during cross-validation: %f' % best_val)
```

建议这里可以把learning rate和regularization_strengths开个大一点的区间，多试试。

```

# 想知道算法是如何运作的很重要的方法是把它的分类错误可视化。在这里的可视化中，我们
# 展示了我们系统错误分类的图片。比如第一列展示的是实际label是飞机，但是被我们系统误
# 标注成其它label的图片。
examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()
```
### Inline question 1:

描述一下你模型分错类的结果。他们看起来有合理吗？

Answer: 笔者表示看起来不怎么合理....LOL。

## 用神经网络训练图片特征


在前面的作业中我们看到直接在像素上训练两层的神经网络结果比线性分类器的效果好。在这个notebook中我们已经看到线性分类器在图片特征上比在像素上效果好。
为了完整性，我们应该试试在图片特征上用神经网络。这个方法应该比前面的方法都好：你应该在测试集上很容易得到一个55%的分类结果。我们最好的模型达到了60%的准确率。
```
from cs231n.classifiers.neural_net import TwoLayerNet

input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

net = TwoLayerNet(input_dim, hidden_dim, num_classes)
best_net = None

################################################################################
# TODO: Train a two-layer neural network on image features. You may want to    #
# cross-validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################
learning_rates = [1e-2 ,1e-1, 5e-1, 1, 5]
regularization_strengths = [1e-3, 5e-3, 1e-2, 1e-1, 0.5, 1]

for lr in learning_rates:
    for reg in regularization_strengths:
        net = TwoLayerNet(input_dim, hidden_dim, num_classes)
        # Train the network
        stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
        num_iters=1500, batch_size=200,
        learning_rate=lr, learning_rate_decay=0.95,
        reg= reg, verbose=False)
        val_acc = (net.predict(X_val_feats) == y_val).mean()
        if val_acc > best_val:
            best_val = val_acc
            best_net = net         
        results[(lr,reg)] = val_acc

# Print out results.
for lr, reg in sorted(results):
    val_acc = results[(lr, reg)]
    print 'lr %e reg %e val accuracy: %f' % (
                lr, reg,  val_acc)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val
#pass
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
```
这部分基本上复用前面的代码就可以了，参数还是要多调试。

## Bonus: Design your own features!

*你已经看到了简单的图片特征可以提升分类效果。到目前为止我们尝试了HOG和颜色直方图，但是其他类型的特征也许能得到更好的分类效果。设计一个新的特征。把它用在CIFAR-10上。解释你的特征是如何运作的，你为什么觉得它会对图像分类有用。在这个notebook中实现它，使用交叉验证来调整超参数，并和HOG特征+颜色直方图的特征的baseline做比较。*

## Bonus: Do something extra!

用这个作业中的资料和代码做点有趣的事。是否有我们应该问的其他问题？你在做这个作业的时候有其他很棒的想法吗？做做吧！

这里可以验证上面提到过的把HOG特征和颜色直方图的特征单独拿出来效果是否变差。








