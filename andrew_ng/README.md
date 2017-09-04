# Andrew Ng
https://www.coursera.org/learn/machine-learning

![](./images/X.png)

# Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like.

## Linear Regression
map input variables to some continuous function

### Hypothesis Function
![](./images/hypothesis_function.png)

Vector:
```
H(x) = X * θ'
```

### Cost Function
![](./images/cost_function.png)
![](./images/cost_function_plot.png)

Vector:
```
h = X * theta;
J = (1 / (2*m)) * sum((h - y) .^ 2);
# J = (1/(2*m)) * ((h-y)' * (h - y));
```

### Gradient Descent
![](./images/gradient_descent.png)
![](./images/gradient_descent_linear.png)

Vector:
```
h = X * theta;
theta = theta - X' * ((alpha / m) * (h - y));
grad = (1.0 ./ m) * X' * (h - y)
```

## Normal Equation
![](./images/normal_equation.png)
![](./images/compare.png)


## Classification
map input variables into discrete categories.

### Hypothesis Function
![](./images/classification.png)

### Decision Boundary
```
h(x) = g(z)
z > 0
Xθ > 0
```
![](./images/decision_boundary.png)

### Cost Function
![](./images/cost_function_classification.png)

### Gradient Descent
![](./images/gradient_descent_classification.png)

## Overfitting

### Linear
Cost Function:
![](./images/overfitting_linear_cf.png)

Gradient Descent:
![](./images/overfitting_linear_gd.png)

Normal Equation:
![](./images/overfitting_ne.png)

### Logistic

Cost Function:
![](./images/overfitting_logistic_cf.png)

Gradient Descent:
![](./images/overfitting_logistic_gd.png)

# Neural Networks
Flow:
![](./images/nn_flow_1.png)
Theta:
![](./images/nn_theta_1.png)
Flow:
![](./images/nn_flow_2.png)
Theta:
![](./images/nn_theta_2.png)

![](./images/nn_d.png)

# Unsupervised Learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like


