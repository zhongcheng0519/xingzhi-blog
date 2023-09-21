+++
author = "行之"
title = "libSVM推理的嵌入式移植"
date = "2023-09-19"
description = "Guide to emoji usage in Hugo"
tags = [
    "Machine Learning",
]
+++

2020年，我主要在为防老人摔倒的产品做算法。我们将一个IMU传感器固定在一根腰带上，从而快速采集人体的6轴数据（三轴加速度数据与三轴角速度数据）。然后使用STM32F4系列的芯片将采集到的数据喂给SVM算法，最终得出是否有摔倒风险的分类结果；另外，我们也在同样的数据上，训练了另外一个分类器，用来判定用户当前的姿态。同样，也是用的SVM。

SVM(Support Vector Machine, 支持向量机)这个算法也是相当经典的一个分类算法了。在深度神经网络出现之前，SVM的表现要好于MLP。而且它的前向推理也很简单，完全可以在低功耗的MCU上进行推理。我们在训练时，使用的是台湾大学林智仁(Lin Chih-Jen)教授开发的[libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)库。但是在准备将libSVM的推理部分移植到STM32上时，遇到了一些挑战。

相比于整个libSVM开源库，其推理部分相对简单，核心公式如下：

$$
\operatorname{sgn}\left(\boldsymbol{w}^{T} \phi(\boldsymbol{x})+b\right)=\operatorname{sgn}\left(\sum_{i=1}^{l} y_{i} \alpha_{i} K\left(\boldsymbol{x}_{i}, \boldsymbol{x}\right)+b\right)
$$

如果想要了解公式的具体含义，可以参考这篇[文章](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)。

多么简单清晰的一个公式，唯一需要变换的就只是核函数而已，而文章中也已经给出了不同的核函数的写法。

另一方面，libsvm训练好后会保存为一个模型文件。而进行libsvm推理时，需要先加载这个文件。而嵌入式的裸机开发中，是没有文件系统的，即使可以加上，也很麻烦。所以，我索性就干了这么个事：自己解析libsvm生成的模型文件，并生成对应的model.h和model.c文件。把**数据**和**用于推理的方法**全部都放在model.h和model.c中，这样，就可以非常简单地在外部调用svm的推理方法了。

我把这个工作放到了我的[github](https://github.com/zhongcheng0519/libsvm_for_embeded)中。实现起来其实非常简单，但是却大量地缩小了嵌入式系统中存储的使用，免去了文件系统的搭建。如果你也有同样的需求，也可以试试看。