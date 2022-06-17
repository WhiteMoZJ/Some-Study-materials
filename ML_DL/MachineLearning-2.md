# 机器学习 Machine Learning - 2

**逻辑回归 Logistic Regression**

## 分类和表示 Classification and Representation

### 分类 Classification

$$
y \in \{0,1,\dots\}
$$

要尝试分类，一种方法是使用线性回归，并将所有大于0.5的预测映射为1，并将所有小于0.5的预测映射为0

分类问题就像回归问题一样，只是=现在要预测的值只采用少量离散值

**二元分类问题**，其中 y 只能采用两个值，即0和1

### 假设函数表达式Hypothesis Representation

逻辑回归模型h~θ~(x)
$$
h_\theta(x)=g(\theta^Tx)\newline
g(z)=\frac{1}{1+e^{-z}}\newline
\to h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}
$$
<img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/1WFqZHntEead-BJkoDOYOw_2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function.png?expiry=1645056000000&hmac=4koaJXtN-NEkGPCmkbTt8I6mBh6UNxZBSbGnaHp1T9Q" alt="img" style="zoom: 80%;" />

g(z)称为**sigmoid函数**或**逻辑函数**

h~θ~(x)用于计算输入为1的**概率(probability)**

如果h~θ~(x) = 0.7 则输出结果为1的概率为70%,换言之输出为0的概率为30%
$$
h_\theta(x)=P(y=1|x;\theta)=1-P(y=0|x;\theta)\newline
P(y=1|x;\theta)+P(y=0|x;\theta)=1
$$

### 决策边界 Decision Boundary

要做出输出0或1，将假设函数设置如下
$$
h_\theta(x)\ge0.5\to y=1\newline
h_\theta(x)\le0.5\to y=0
$$
带入到逻辑函数得
$$
g(z)\ge0.5\newline
when\space z\ge0
$$
如果输入值是 θ^T^X
$$
h_\theta(x)=g(\theta^Tx)\ge0.5\newline
when\space\theta^Tx\ge0
$$
得出
$$
\space\theta^Tx\ge0\to y=1\newline
\space\theta^Tx\le0\to y=0
$$
**决策边界**是分隔y = 0且y = 1的区域的线，是由假设函数创建的

**Example**
$$
\theta=
\begin{bmatrix}
{5}\\{-1}\\{0}
\end{bmatrix}
\newline
y=1\space{if}\space5+(-1)x_1+0x_2\ge0\newline
5-x_1\ge0\newline
x_1\le5
$$
x~1~ = 5是决策边界，左边结果输出为y = 1，右边输出为y = 0

g(z)不一定是线性的

## 逻辑回归模型 Logistic Regression Model

### 代价函数 Cost Function

逻辑回归的代价函数和线性回归不同，不能使用与线性回归相同的成本函数，因为逻辑函数将导致输出呈**波浪形**，从而**导致许多局部最优**

逻辑回归的代价函数
$$
J(\theta)=\frac{1}{m}\sum^{m}_{i=1}\operatorname{Cost}(h_\theta(x^{(i)}),y^{(i)})\newline
\operatorname{Cost}(h_\theta(x),y)=-\log(h_\theta(x))\space\operatorname{if}\space y=1\newline
\operatorname{Cost}(h_\theta(x),y)=-\log(1-h_\theta(x))\space\operatorname{if}\space y=0
$$
当y = 1时，J(θ)与h~θ~(x)的图像为

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Q9sX8nnxEeamDApmnD43Fw_1cb67ecfac77b134606532f5caf98ee4_Logistic_regression_cost_function_positive_class.png?expiry=1645142400000&hmac=1ivwTB37yUCSqsGZAZL0liVCXvQrOYZ1VMRz59X8984)

同样，当y = 0时，J(θ)与h~θ~(x)的图像为

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Ut7vvXnxEead-BJkoDOYOw_f719f2858d78dd66d80c5ec0d8e6b3fa_Logistic_regression_cost_function_negative_class.png?expiry=1645142400000&hmac=BeXcxrD1W6ys-hAr2l8O3uKaYoyGX764udcdIiVWKtg)
$$
\operatorname{Cost}(h_\theta(x),y)=0\space{\operatorname{if}}\space h_\theta(x)=y\newline
\operatorname{Cost}(h_\theta(x),y)\to\infin\space{\operatorname{if}}\space y=0\space{\operatorname{and}}\space h_\theta(x)\to1\newline
\operatorname{Cost}(h_\theta(x),y)\to\infin\space{\operatorname{if}}\space y=1\space{\operatorname{and}}\space h_\theta(x)\to0
$$

#### 简化代价函数 Simplified Cost Function

更简单的表示为
$$
\operatorname{Cost}(h_\theta(x),y)=-y\log h_\theta(x)-(1-y)\log(1-h_\theta(x))
$$

$$
J(\theta)=\frac{1}{m}\sum^{m}_{i=1}\operatorname{Cost}(h_\theta(x^{(i)}),y^{(i)})\newline
=-\frac{1}{m}[\sum^{m}_{i=1}y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]
$$

矢量化实现是
$$
h =g(X\theta)\newline
J(\theta)=\frac{1}{m}(-y^T\log(h)-(1-y)^T\log(1-h))
$$

### 梯度下降 Gradient Descent

$$
J(\theta)=-\frac{1}{m}[\sum^{m}_{i=1}y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]
$$

梯度下降的一般公式：
$$
Repeat
\{\newlineθ_j:=θ_j−α\frac{∂}{∂θ_j}J(θ)
\newline\}
$$
计算导数部分：
$$
Repeat
\{\newlineθ_j:=θ_j−\frac{\alpha}{m}\sum^m_{i=1}(h_\theta(x^{(i)}-y^{(1)}))x_j^{(i)}
\newline\}
$$
此算法与线性回归中使用的算法相同，仍然需要**同时更新θ中的所有值**

矢量化表示为：
$$
\theta:=\theta-\frac{\alpha}{m}X^T(g(X\theta)-\vec{y})
$$

### 高级优化 Advanced Optimization

首先需要提供一个函数，用于计算给定输入值θ的以下两个函数
$$
J(\theta)\newline
\frac{\partial}{\partial\theta_j}J(\theta)
$$
编写一个函数来返回以下两个：

```matlab
function [jVal, gradient] = costFunction(theta)
	jVal = [...code to compute J(theta)...];
	gradient = [...code to compute derivative of J(theta)...];
end
```

然后，可以使用`fminunc()`优化算法以及`optimset()`函数，该函数创建一个包含要发送到`fminunc()`的选项的对象（注意：MaxIter 的值应该是整数，而不是字符串 ）

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
	[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

给函数`fminunc()`成本函数，θ值的初始向量，以及事先创建的"options"对象

## 多类别分类 Multiclass Classification

### 一对多 One-vs-all

$$
y\in\{0,1,\dots,n\}
$$

将问题划分为n+1（因为索引从 0 开始）二元分类问题;在每一个中，预测"y"是一个类的成员的概率
$$
y\in\{0,1,\dots,n\}\newline
h_\theta^{(0)}(x)=P(y=0|x;\theta)\newline
h_\theta^{(1)}(x)=P(y=1|x;\theta)\newline
\dots\newline
h_\theta^{(n)}(x)=P(y=n|x;\theta)\newline
\operatorname{prediction}=\max_i(h_\theta^{(i)}(x))
$$
基本上是选择一个类，然后将所有其他类集中到一个第二类中。反复这样做，将二元逻辑回归应用于每个情况，然后使用返回最高值的假设作为预测

训练逻辑回归分类器h~θ~(x)对于每个类，用于预测 y = i 的概率。

要对新 x 进行预测，请选择最大化的类h~θ~(x)

## 过度拟合 Overfitting

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/0cOOdKsMEeaCrQqTpeD5ng_2a806eb8d988461f716f4799915ab779_Screenshot-2016-11-15-00.23.30.png?expiry=1645574400000&hmac=t9fmQYHgf_0cUKnKzo_bCJ_VGw-F-HqP3VTjmQb4fUU)

**欠拟合**或**高偏差**是指我们的假设函数h的形式与数据趋势的映射较差，通常是由功能太简单或使用太少的功能引起的

在另一个极端，**过拟合**或**高方差**是由拟合可用数据但不能很好地泛化以预测新数据的假设函数引起的，通常是由一个复杂的函数引起的，该函数创建了许多与数据无关的不必要的曲线和角度

有两个主要选项可以解决过度拟合问题：

减少功能数量：

- 手动选择要保留的功能
- 使用模型选择算法

正则化：

- 保留所有特征，但减少参数的大小θ~j~
- 当我们有很多稍微有用的功能时，正则化效果很好

### 代价函数 Cost Function

如果从假设函数中获得过拟合，可以通过增加其成本来减少函数中某些项所承载的权重

使以下函数更二次化
$$
\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4
$$
消除θ~3~x^3^和θ~4~x^4^.实际上，在不摆脱这些特征或改变假设形式的情况下，可以**修改成本函数**
$$
\min_\theta\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+1000·\theta_3^2+1000·\theta_4^2
$$
最后添加两个条件，以增加成本θ~3~x^3^和θ~4~x^4^

为了使成本函数接近于零，不得不减少θ~3~x^3^和θ~4~x^4^接近于零

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/j0X9h6tUEeawbAp5ByfpEg_ea3e85af4056c56fa704547770da65a6_Screenshot-2016-11-15-08.53.32.png?expiry=1645574400000&hmac=YH4gFjALjFOLwprtOYS59MBj_xv7A6Ylm2YMbo8eP7I)

可以在单个求和中将所有θ参数正则化为：
$$
\min_\theta\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{i=1}^n\theta^2_j
$$
λ：**正则化参数**，决定了θ参数的成本被扩大了多少

使用上述成本函数和额外求和，可以平滑假设函数的输出以减少过拟合。如果选择λ过大，则可能会使函数过于平滑，并导致欠拟合
