# AutoML in Action

注意：使用 pyctr 环境。

## 版本

只列举比较关键的包

+ python 3.6.8
+ tensorflow 1.14.0
+ numpy 1.16.4
+ pandas 0.24.2
+ scikit-learn 0.21.2
+ scipy 1.3.0
+ request 2.22.0
+ keras 2.2.4
+ graphviz 0.11
+ adanet 0.6.2
+ tensorflowonspark 1.4.3
+ pyspark 2.4.3

可视化需要安装 [Graphiz](https://graphviz.gitlab.io/download/)

`brew install graphviz`

Spark 版本 2.4.3

## AutoML 库

+ autoKeras 0.4.0
+ nni 0.8
+ adaNet 0.6.2

## Notes

One use-case that has worked for us at Google, has been to take a production model's TensorFlow code, convert it to into an adanet.subnetwork.Builder, and adaptively grow it into an ensemble. In many cases, this has given significant performance improvements.

### Simple DNN

creates two candidate fully-connected neural networks at each iteration with the same width, but one an additional hidden layer. To make our generator adaptive, each subnetwork will have at least the same number of hidden layers as the most recently added subnetwork to the previous_ensemble.