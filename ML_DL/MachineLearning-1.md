# 机器学习 Machine Leaning - 1

**线性回归 Linear Regression**

## 介绍

### 定义 Definition

- Arthur Samuel: "the field of study that gives computers the ability to learn **without being explicitly programmed**."

- Tom Mitchell: "A computer program is said to learn from experience E with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in T, as measured by P, improves with **experience E**."

分类:

- Supervised learning
- Unsupervised learning

### 监督学习 Supervised learning

给出一个数据并且知道输出应该是什么样的，也就是知道输入与输出的关系

监督学习分为**“回归regression”**和**“分类classification”**问题

**回归**

预测**连续输出continuous output**的结果，将输入的变量映射到某个连续函数中

Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output

Regression - Given a picture of a person, we have to predict their age on the basis of the given picture

**分类**

预测**离散输出discrete output**的结果，将输入的变量映射到离散类别中

Whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign. 

### 无监督学习 Unsupervised Learning

在很少或者根本不知道结果应该是什么样子的情况下解决问题，**从数据中推导出结构（derive structure from data）** ，但不一定知道变量的影响

通过**聚类数据（clustering the data）**基于数据中变量之间的关系推到此结构

‎对于无监督学习，没有基于**预测结果prediction results**的**反馈feedback**

> ‎**聚类**：收集1,000,000个不同的基因，并找到一种方法，将这些基因自动分组为通过不同变量（例如寿命，位置，角色等）以某种方式相似或相关的组

## 单变量线性回归 Linear Regression with One Variable

### 模型表示 Model Representation

- x^(i)^表示“输入”变量，也叫输入要素
- y^(i)^表示“输出”或者要预测的目标变量
- (x^(i)^,y^(i)^)称为**训练示例**
- m个训练示例的列表(x(i),y(i));i=1,...,m 称为**训练集**

给定一个训练集，学习一个函数h：X→Y，这样h(x)是y的相应值的"好"预测因子，函数h被称为**假设hypothesis**

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/H6qTdZmYEeaagxL7xdFKxA_2f0f671110e8f7446bb2b5b2f75a8874_Screenshot-2016-10-23-20.14.58.png?expiry=1643500800000&hmac=1BN7krQYUI-3acEPY6Qd3xaKnMBGiFX_ESnlQAauzoI)

预测的目标变量是连续的时，学习问题称为回归问题；当y只能取少量离散值时，称之为分类问题



### 代价函数 Cost Function

对于一个假设的回归函数h为
$$
h_θ(x)=θ_0+θ_1x
$$
参数为θ~0~, θ~1~

通过使用代价函数来测量假设**函数**的准确性
$$
J(θ_0,θ_1)=\frac{1}{2m}\sum_{i=1}^m(h_θ(x_i)−y_i)^2
$$
把它拆开就是
$$
\frac{1}{2}\overline{x}
$$
预测值与实际值之间的差值平方的平均值

此函数也称为**"平方误差函数(Squared error function)"**或**"均方误差(Mean squared error)"**

MATLAB

```matlab
function J = computeCost(X, y, theta)
m = length(y); % number of training examples
J = 0;
v = X * theta - y;
v = v .^ 2;
J = 1 / (2 * m) * sum(v);
end
```

**平均值为1/2减半**，以方便计算**梯度下降(gradient descent)**，因为平方函数的导数项将抵消1/2项

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/R2YF5Lj3EeajLxLfjQiSjg_110c901f58043f995a35b31431935290_Screen-Shot-2016-12-02-at-5.23.31-PM.png?expiry=1643500800000&hmac=tY1WkPCQstKi6NLeu4ac7h22IWVFOsvGsjz9Vt1QfmA)

最终**使J(θ~0~,θ~1~)最小 (minimize the cost function)**
$$
\operatorname{minimize}J(θ_0,θ_1)
$$
简化，使θ~0~ = 0：

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/_B8TJZtREea33w76dwnDIg_3e3d4433e32478f8df446d0b6da26c27_Screenshot-2016-10-26-00.57.56.png?expiry=1643500800000&hmac=Gx0Q_s1YJ7ABnRpZNAB5Sc7pffcNqok3qe82w2NVSEE)

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8guexptSEeanbxIMvDC87g_3d86874dfd37b8e3c53c9f6cfa94676c_Screenshot-2016-10-26-01.03.07.png?expiry=1643500800000&hmac=RXr9E_cndSe9FdtBSyvkBESrU6TX8UP6taZaoFFee8M)

*如果θ~0~,θ~1~都有值的话：

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/N2oKYp2wEeaVChLw2Vaaug_d4d1c5b1c90578b32a6672e3b7e4b3a4_Screenshot-2016-10-29-01.14.37.png?expiry=1643500800000&hmac=O8_4h0c530W9leFOsinEm13FsccWHFix_32TZXV7MBw)

MATLAB

```matlab
function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples
J = 0;

J = 1/(2*m)*(X*theta - y)'*(X*theta - y);

end
```



### 梯度下降 Gradient Descent

**用于估计假设函数中的参数**

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bn9SyaDIEeav5QpTGIv-Pg_0d06dca3d225f3de8b5a4a7e92254153_Screenshot-2016-11-01-23.48.26.png?expiry=1643760000000&hmac=VgfkoSmd15lAH6tu3Nx9k3_xII_6gLTrvS46h3MIHVo)

成本函数位于图中坑的最底部时，即当其值为最小值时。这样做的方法是取成本函数的导数（函数的切线）。切线的斜率是该点的导数，它将提供前进的方向。向下降最陡的方向降低成本函数。每个步骤的大小由参数α确定，称为**学习速率（Learning Rate）**

上图中每个"星星"之间的距离表示由我们的参数α确定的步长。较小的α将导致较小的步骤，较大的α会导致较大的步骤。采取步骤的方向由J*(*θ~0~,θ~1~).根据图表上的起点，可能会在不同的点结束。上图显示了两个不同的起点，最终在两个不同的地方结束

梯度下降算法为：

**重复直到收敛（repeat until convergence）{**
$$
θ_j:=θ_j-α\frac{∂}{∂θ_j}J(θ_0,θ_1),j=0,1
$$
**}**

MATLAB

```matlab
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    v = X' * (X * theta - y);
    theta = theta - alpha * (1/m) * v;
    
    % Save the cost J in every iteration  
    J_history(iter) = computeCost(X, y, theta);
end
end
```

**在每次迭代j时，应该同时更新参数**。在计算另一个参数之前更新特定参数j(th)迭代将产生错误

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/yr-D1aDMEeai9RKvXdDYag_627e5ab52d5ff941c0fcc741c2b162a0_Screenshot-2016-11-02-00.19.56.png?expiry=1643760000000&hmac=1qCIuK3tS4u2wBevH4TMjmuTe1ot_QlR5C_Zztavz1Q)

**应该调整参数α以确保梯度下降算法在合理的时间内收敛**。未能收敛或获得最小值的时间过长意味着步长是错误的

当我们接近凸函数的底部时，导数将始终为0，因此我们得到：

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/RDcJ-KGXEeaVChLw2Vaaug_cb782d34d272321e88f202940c36afe9_Screenshot-2016-11-03-00.06.00.png?expiry=1643846400000&hmac=loqegz5HkWRyCrH0mWEB4f18U3XhZmVQNV8D2WGfv5I)

### 线性回归的梯度下降 Gradient Descent for Linear Regression

当专门应用于线性回归的情况时，可以推导出梯度下降方程的新形式
$$
\operatorname{repeat\space until\space convergence:}{\{}\newline
\theta_0:=\theta_0-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x_i)-y_i)\newline
\theta_1:=\theta_1-\alpha\frac{1}{m}\sum_{i=1}^{m}((h_{\theta}(x_i)-y_i)x_i)\newline
{\}}
$$
m是训练集的大小，θ~0~一个将同时变化的常量θ1和xi, yi是给定训练集（数据）的值

如果我们从对我们的假设的猜测开始，然后反复应用这些梯度下降方程，我们的假设将变得越来越准确

在每个步骤上查看整个训练集中的每个示例，称为**批处理梯度下降batch gradient descent**

## 多变量线性回归 Multivariate Linear Regression

### 多特征 Multiple Features

**x~j~^(i)^ = 特征j在ith的训练示例**
**x^(i)^ = ith训练示例的输入（特征）**
**m = 训练示例的数量**
**n = 特征数**

容纳这些**多个特征的假设函数（多假设函数，multivariable hypothesis function）**的多变量形式如下
$$
h_\theta = \theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3+\cdots+\theta_nx_n
$$
用矩阵表示
$$
h_\theta(x)=
\begin{bmatrix}
{\theta_{0}}&{\theta_{1}}&{\cdots}&{\theta_{n}}
\end{bmatrix}
\begin{bmatrix}
{x_{0}}\\{x_{1}}\\{\vdots}\\{x_{n}}
\end{bmatrix}
=\theta^Tx
$$
这是一个训练示例的假设函数的**矢量化vectorization**

### 多变量梯度下降 Gradient Descent for Multiple Variables

$$
\operatorname{repeat\space until\space convergence:}{\{}
\newline
\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)}-y^{(i)})·x_j^{(i)}{\space\space\space\space}\operatorname{for{\space}j:=0{\cdots}n}
\newline
{\}}
$$

MATLAB

```matlab
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    theta = theta - alpha / m * (X' * (X * theta - y));
    J_history(iter) = computeCostMulti(X, y, theta);

end
end
```



#### 特征缩放 Feature Scaling

通过将每个输入值置于大致相同的范围内来加速梯度下降。这是因为θ在小范围内会迅速下降，在大范围内会缓慢下降，因此当变量非常不均匀时，会低效地振荡到最佳值

理想：**−1 ≤x^(i)^≤ 1** 或 **−0.5 ≤x^(i)^≤ 0.5**

**特征缩放**涉及将输入值除以输入变量的范围（即最大值减去最小值），从而生成仅 1 的新范围。**均值归一化**涉及从输入变量的值中减去输入变量的平均值，从而为输入变量生成新的平均值，即零

公式如下：
$$
x_i:=\frac{x_i-\mu_i}{s_i}
$$
**μ~i~**是所有特征的**平均值**，**s~i~**是所有特征的范围（最大值-最小值）或是**标准偏差（standard deviation）**

MATLAB

```matlab
function [X_norm, mu, sigma] = featureNormalize(X)
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
      
m = size(X, 1);
mu = mean(X, 1); 
sigma = std(X);  
for i = 1 : m
    X_norm(i, :) = (X(i, :) - mu) ./ sigma;
end
end
```



#### 学习率 Learning Rate

$$
θ_j:=θ_j-α\frac{∂}{∂θ_j}J(θ)
$$

**调试梯度下降 Debugging gradient descent**:在 x 轴上创建具有迭代次数(number of iterations)的图。现在绘制成本函数 J(θ)在梯度下降的迭代次数上。如果 J(θ)增加，那么你可能需要减小α

**自动收敛测试 Automatic convergence test:**如果 J(θ)在一次迭代中减少小于E，则声明收敛，其中E是一些小值，例如10^−3^.但是，在实践中很难选择此阈值

<img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/FEfS3aajEea3qApInhZCFg_6be025f7ad145eb0974b244a7f5b3f59_Screenshot-2016-11-09-09.35.59.png?expiry=1644537600000&hmac=DzxmglhxOa_CbOgbkiJHumDESQyrr9do9xQWnSVBKHA" alt="img" style="zoom:80%;" />

总结：

**如果*α*过小：收敛缓慢**

**如果*α*过大：可能不会再每次迭代中减小，因此可能不会收敛**

### 特征和多项式回归 Features and Polynomial Regression

假设函数不必是线性的（直线），如果它不能很好地拟合数据。

我们可以通过将其设置为二次，三次或平方根函数（或任何其他形式）来改变假设函数的**行为或曲线**

例如，如果我们的假设函数是
$$
h_\theta(x) = \theta_0 + \theta_1 x_1
$$
然后我们可以基于以下条件创建其他功能x~1~，获取二次函数
$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2
$$
或三次函数
$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3x_1^2
$$
要使其成为平方根函数
$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 \sqrt{x_1}
$$
如果以这种方式选择功能，那么功能缩放将变得非常重要

如果x~1~范围为1 - 1000，然后范围为x~1~^2^变为1 - 1000000，x~1~^3^变为1 - 1000000000

### 正规方程 Normal Equation

在"正规方程"方法中，将通过显式取其相对于θ~j~的导数，并将它们设置为零来最小化J，使能够在没有迭代的情况下找到最佳的θ
$$
\theta =(X^TX)^{-1}X^Ty
$$
m为训练示例的数量，n为特征数，**X的形状为m*(n+1)，y为m维的向量(m-dim vector)**

MATLAB

```matlab
function [theta] = normalEqn(X, y)
theta = zeros(size(X, 2), 1);
theta = pinv(X'*X)*X'*y;
end
```

如下

<img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/dykma6dwEea3qApInhZCFg_333df5f11086fee19c4fb81bc34d5125_Screenshot-2016-11-10-10.06.16.png?expiry=1644537600000&hmac=2Kv05Zglw8ipICLx-XsDxdMaam_o-GyHxrv7za8y_IA" alt="img" style="zoom:80%;" />

**不需要**进行特征缩放

梯度下降与正规方程的比较

| 梯度下降 Gradient Descent | 正规方程 Normal Equation |
| :-----------------------: | :----------------------: |
|         需要选择α         |       不需要选择α        |
|       需要很多迭代        |        不需要迭代        |
|      复杂度O(kn^2^)       |      复杂度O(n^3^)       |
|  当n非常大时也能很好运行  |    当n非常大时会较慢     |

如果我们有非常多的特征，则正态方程将很慢
在实践中，**当 n 超过10,000时**，可能是从正常解到迭代过程的好时机

#### 不可逆性 Non-invertibility*

如果X^T^X是**不可逆的**，常见原因可能有：

- 冗余特征，其中两个特征非常密切相关（即它们是线性依赖的）
- 特征过多（例如.m ≤ n）在这种情况下，请删除某些功能或使用"正则化 regularization"

上述问题的解决方案包括删除与另一个要素线性依赖的要素，或者在要素过多时删除一个或多个要素
