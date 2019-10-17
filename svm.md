# 参考文档

# 了解SVM
## Logistic回归
 给定一些数据点，它们分别属于两个不同的类，现在要找到一个线性分类器把这些数据分成两类。如果用x表示数据点，用y表示类别（y可以取1或者-1，分别代表两个不同的类, 一个线性分类器的学习目标便是要在n维的数据空间中找到一个超平面（hyper plane），这个超平面的方程可以表示为

> <span style="color:red">[1]</span> $w^{T}x + b = 0$

## Cost Function
https://yoyoyohamapi.gitbooks.io/mit-ml/content/SVM/articles/%E4%BB%A3%E4%BB%B7%E5%87%BD%E6%95%B0.html

> <span style="color:red">[2]</span>
> $$
> \min_{\theta} \frac{1}{m}[\sum_{i=1}^{n}y^{i}(-\log h_{\theta}(x^{i})) + (1-y^{i})(1-\log(1-h_{\theta}(x^{i})))] + \frac{1}{2}\sum_{j=1}^{n}\theta^{2}_{j}
> $$
> $$h_{\theta}(x) = \frac{1}{1+\exp^{-\theta^{T}x}}$$

## Large Margin Classifier

### 点到直线的距离

$(x_{0}, y_{0})$ 到 $Ax + By + C = 0$ 距离可用如下公式表示: $d = \frac{|Ax_{0} + By_{0} + C|}{\sqrt{A^{2} + B^{2}}}$

then:
$(x_{0}, y_{0})$ 到 $w^{T}X + b = 0$  距离可用如下公式表示: $d = \frac{|wx_{0} + b|}{\sqrt{A^{2} + B^{2}}}$

### Margin

1. 为了预测精度足够高, y=1时, 希望θTx≥1, y=0时, 希望θTx≤−1
2. margin线上的点$(x_{0}, y_{0})$ 到直线 $w^{T}X + b = 0$ 的距离为 <span style="color:red">[3]</span>$d = \frac{1}{||w||}$, 因为margin线为$w^{T}x + b = 1$或$w^{T}x + b = -1$
3. 将margin线加入y, 即$y({w^{T}x + b}) = -1$

### 问题求解变为
<span style="color:red">[4]</span>
$$\max\frac{1}{||w||} $$
$$s.t. y_{i}(w^{T}x_{i} + b) \geq 1$$
$$ i = 1,...,n$$

# 深入SVM

## 从线性可分到线性不可分
由于求的最大值相当于求的最小值，所以上述目标函数等价于<span style="color:red">[5]</span>

$$\min\frac{1}{2}||w||^2 $$
$$s.t. y_{i}(w^{T}x_{i} + b) \geq 1$$
$$ i = 1,...,n$$

## 最优解问题分类
1. 无约束问题优化，可通过求导
2. 有等式约束优化，可通过拉格朗日乘子法
3. 有不等式约束优化，可通过拉格朗日对偶性解决，满足KKT条件

## 拉格朗日对偶性

我们的问题符合第三类最优解问题:
$$
\ell(w, b, \alpha) = \frac{1}{2}||w||^{2} - \sum_{i=1}^{n}\alpha_{i}(y_{i}(w^{T}x_{(i)} + b) - 1)
$$

令:
$$
\psi(w) = \max_{\alpha_{i} \geq 0}\ell(w, b, \alpha)
$$

$\Rightarrow$

$$
\psi(w) = \min_{w, b}\max_{\alpha_{i} \geq 0}\ell(w, b, \alpha)
$$

有兰格朗日对偶性:
$$
P^{*} = \min_{w, b}\max_{\alpha_{i} \geq 0}\ell(w, b, \alpha)
$$
$$
D^{*} = \max_{\alpha_{i} \geq 0}\min_{w, b}\ell(w, b, \alpha)
$$
$$
D^{*} \leq P^{*}
$$
在满足KKT条件，这两者相等，这个时候就可以通过求解对偶问题来间接地求解原始问题, KKT条件:
$$
optimize f(x)
$$
$
s.t.
$
$$
g(x) \leq 0
$$
$$
h(x) \equiv 0
$$
$\Rightarrow$
$$
\ell(x, \alpha, \lambda) = f(x) + \alpha g(x) + \lambda h(x)
$$
$
s.t.
$

$$
a_{i} \geq 0, g(x) \leq 0, \sum_{i=1}^{n}\alpha_{i}g(x_{i}) = 0, \lambda \neq 0
$$

此问题可变为:

<span style="color:red">[6]</span>
$$
\ell(w, b, \alpha) = \frac{1}{2}||w||^{2} + \sum_{i=1}^{n}\alpha_{i}(1- y_{i}(w^{T}x_{(i)} + b))
$$
$$
a_{i} \geq 0, g(x)=1- y_{i}(w^{T}x_{(i)} + b) \leq 0, \sum_{i=1}^{n}\alpha_{i}g(x_{i}) = 0
$$

$\ell(w, b, \alpha)$有最大值，因为$a_{i} \geq 0, g(x) \leq 0$, 分别对$w, b$求导:

$\frac{\partial \ell}{\partial w} = 0$ $\Rightarrow$ $w = \sum_{i=1}^{n}\alpha_{i}y_{i}x_{i}$

$\frac{\partial \ell}{\partial b} = 0$ $\Rightarrow$ $\sum_{i=1}^{n}\alpha_{i}y_{i} = 0$

带入方程，$\Rightarrow$

$\ell(w, b, \alpha) = \frac{1}{2}||w||^{2} + \sum_{i=1}^{n}\alpha_{i}(1- y_{i}(w^{T}x_{(i)} + b))$

$\ell(w, b, \alpha) = \frac{1}{2}w^{T}w + \sum_{i=1}^{n}\alpha_{i} - \sum_{i=1}^{n}\alpha_{i}y_{i}w^{t}x_{i} - b \sum_{i=1}^{n}\alpha_{i}y_{i}$

$\ell(w, b, \alpha) = \frac{1}{2}w^{T}w + \sum_{i=1}^{n}\alpha_{i} - \sum_{i=1}^{n}\alpha_{i}y_{i}w^{T}x_{i}$

$\ell(w, b, \alpha) = \frac{1}{2}(\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i})^{T}(\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i}) + \sum_{i=1}^{n}\alpha_{i} - \sum_{i=1}^{n}\alpha_{i}y_{i}(\sum_{j=1}^{n}\alpha_{j}y_{j}x_{j})^{T}x_{i}$

$\ell(w, b, \alpha) = \frac{1}{2}\sum_{i=1,j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j} + \sum_{i=1}^{n}\alpha_{i} - \sum_{i=1,j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}$

$\ell(w, b, \alpha) = \sum_{i=1}^{n}\alpha_{i} - \frac{1}{2}\sum_{i=1,j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}$

当$\ell(w, b, \alpha)$取到极值时<span style="color:red">[7]</span>:

$\ell(w, b, \alpha) = \sum_{i=1}^{n}\alpha_{i} - \frac{1}{2}\sum_{i=1,j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j} = f(x)$

## 核函数(mercer定理，半正定)
分类:
1. 线性核函数 $K(x_{i},x_{j}) = x_{i}^Tx_{j}$
2. 多项式核函数 $K(x_{i},x_{j}) = (\gamma_{i}^Tx_{j} + b)$
3. 高斯核函数 $K(x_{i},x_{j}) = \exp(-\gamma||x_{i} - x_{j}||^{2})$
4. 拉格拉斯核函数
5. sigmod核函数

假设:
$$
x =
\begin{bmatrix}
x_{1} \\
x_{2}
\end{bmatrix}
$$
映射到
$$
\phi =
\begin{bmatrix}
x_{1}x_{1} \\
x_{1}x_{2} \\
x_{2}x_{1} \\
x_{2}x_{2}
\end{bmatrix}
$$

$\Rightarrow$

$
K(\phi(m), \phi(n)) = \phi(m)^{T}\phi(n)
$

$
K(m, n) = m_{1}m_{1}n_{1}n_{1} + m_{1}m_{2}n_{1}n_{2} + m_{2}m_{1}n_{2}n_{1} + m_{2}m_{2}n_{2}n_{2}
$

$
K(m, n) = m_{1}m_{1}n_{1}n_{1} + 2m_{1}m_{2}n_{1}n_{2} + m_{2}m_{2}n_{2}n_{2}
$

$
K(m, n) = (m_{1}n_{1} + m_{2}n_{2})^2
$

$
K(m, n) = (m^Tn)^2
$

# SMO
取$\alpha_{1}, \alpha_{2}$作为参数

$
D^{*} = \max_{\alpha_{i} \geq 0}\min_{w, b}\ell(w, b, \alpha)
$

$\ell(w, b, \alpha) = \sum_{i=1}^{n}\alpha_{i} - \frac{1}{2}\sum_{i=1,j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j} = f(x)$

$\ell(w, b, \alpha) = \alpha_{1} + \alpha_{2} + \sum_{3}^{n}\alpha_{i} -\frac{1}{2}[\alpha_{1}y_{1}\sum_{j=1}^{n}\alpha_{j}y_{j}(x_{1}^Tx_{j}) + \alpha_{2}y_{2}\sum_{j=1}^{n}\alpha_{j}y_{j}(x_{2}^Tx_{j}) + \sum_{i=3,j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}(x_{i}^Tx_{j})] + \sum_{i=3}^{n}\alpha_{i}$

$\ell(w, b, \alpha) = \alpha_{1} + \alpha_{2} + \sum_{3}^{n}\alpha_{i} -\frac{1}{2}[\alpha_{1}y_{1}\alpha_{1}y_{1}(x_{1}^Tx_{1}) + \alpha_{1}y_{1}\alpha_{2}y_{2}(x_{1}^Tx_{2}) + \alpha_{1}y_{1}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{1}^Tx_{j}) + \alpha_{2}y_{2}\alpha_{1}y_{1}(x_{2}^Tx_{1}) + \alpha_{2}y_{2}\alpha_{2}y_{2}(x_{2}^Tx_{2}) + \alpha_{2}y_{2}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{2}^Tx_{j}) +\sum_{i=3,j=3}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}(x_{i}^Tx_{j})]$

$\ell(w, b, \alpha) = \alpha_{1} + \alpha_{2} - \frac{1}{2}[\alpha_{1}y_{1}\alpha_{1}y_{1}(x_{1}^Tx_{1}) + \alpha_{1}y_{1}\alpha_{2}y_{2}(x_{1}^Tx_{2}) + \alpha_{1}y_{1}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{1}^Tx_{j}) + \alpha_{2}y_{2}\alpha_{1}y_{1}(x_{2}^Tx_{1}) + \alpha_{2}y_{2}\alpha_{2}y_{2}(x_{2}^Tx_{2}) + \alpha_{2}y_{2}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{2}^Tx_{j}] + C$

$\ell(w, b, \alpha) = \alpha_{1} + \alpha_{2} - \frac{1}{2}[\alpha_{1}^2y_{1}^2(x_{1}^Tx_{1}) + \alpha_{1}y_{1}\alpha_{2}y_{2}(x_{1}^Tx_{2}) + \alpha_{1}y_{1}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{1}^Tx_{j}) + \alpha_{2}y_{2}\alpha_{1}y_{1}(x_{2}^Tx_{1}) + \alpha_{2}^2y_{2}^2(x_{2}^Tx_{2}) + \alpha_{2}y_{2}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{2}^Tx_{j}] + C$

由于$y \in \left\{-1, 1\right\}$, 所以$y^2 = 1$, 由于线性核函数$x_{1}^{T}x_{2} = x_{2}^{T}x_{1}$

$\Rightarrow$

$\ell(w, b, \alpha) = \alpha_{1} + \alpha_{2} - \frac{1}{2}[\alpha_{1}^2(x_{1}^Tx_{1}) + 2\alpha_{1}y_{1}\alpha_{2}y_{2}(x_{1}^Tx_{2}) + \alpha_{1}y_{1}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{1}^Tx_{j}) + \alpha_{2}^2(x_{2}^Tx_{2}) + \alpha_{2}y_{2}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{2}^Tx_{j}] + C$

由$\Rightarrow$ $\sum_{i=1}^{n}\alpha_{i}y_{i} = 0$，所以

$\alpha_{1}y_{1} + \alpha_{2}y_{2} + \sum_{i=3}^{n}\alpha_{i}y_{i} = 0$

$\Rightarrow$

$\alpha_{1}y_{1} + \alpha_{2}y_{2} = \zeta$
$\alpha_{1}^{old}y_{1} + \alpha_{2}^{old}y_{2} = \alpha_{1}^{new}y_{1} + \alpha_{2}^{new}y_{2} = \zeta$

$\Rightarrow$

$\alpha_{1}^{old}y_{1}y_{1} + \alpha_{2}^{old}y_{2}y_{1} = \zeta y_{1}$

$\Rightarrow$

$\alpha_{1}^{old} = \zeta y_{1} - \alpha_{2}^{old}y_{2}y_{1}$

$\Rightarrow$

$\alpha_{1}^{old} = \zeta^{'} - s\alpha_{2}^{old}$

$\Rightarrow$

$\ell(w, b, \alpha) = \alpha_{1} + \alpha_{2} - \frac{1}{2}[\alpha_{1}^2(x_{1}^Tx_{1}) + 2\alpha_{1}y_{1}\alpha_{2}y_{2}(x_{1}^Tx_{2}) + \alpha_{1}y_{1}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{1}^Tx_{j}) + \alpha_{2}^2(x_{2}^Tx_{2}) + \alpha_{2}y_{2}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{2}^Tx_{j}] + C$

$\Rightarrow$

$\ell(w, b, \alpha) = (\zeta^{'} - s\alpha_{2}) + \alpha_{2} - \frac{1}{2}[(\zeta^{'} - s\alpha_{2})^2(x_{1}^Tx_{1}) + 2(\zeta^{'} - s\alpha_{2})y_{1}\alpha_{2}y_{2}(x_{1}^Tx_{2}) + (\zeta^{'} - s\alpha_{2})y_{1}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{1}^Tx_{j}) + \alpha_{2}^2(x_{2}^Tx_{2}) + \alpha_{2}y_{2}\sum_{j=3}^{n}\alpha_{j}y_{j}(x_{2}^Tx_{j}] + C$

函数中变量只有$\alpha_{2}$, 求取最大值，需要对 $\alpha_{2}$求偏导数:

$\frac{\partial \ell}{\partial \alpha_{2}} = 0$

$\Rightarrow$

$\alpha_{2} = C^{'}$

$\Rightarrow$

$\alpha_{1} = \zeta^{'} - s\alpha_{2}$

$\Rightarrow$

$\alpha_{2}^{new} = C^{''}$

$\Rightarrow$

$\alpha_{1}^{new} = C^{'''}$
