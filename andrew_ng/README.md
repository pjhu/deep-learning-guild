# Andrew Ng
https://www.coursera.org/learn/machine-learning

![](./images/X.png)

# Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like.

## Linear Regression
map input variables to some continuous function
![](./images/linear.png)

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
```
grad = 1.0 / m * X' * (h -y) + lambda / m * ([0;theta(2:end,:)]);
```

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

## Backpropagation 
https://zh.wikipedia.org/wiki/%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

# Advice for Applying Machine Learning

## Model Selection
break down our dataset into the three sets:
Training set: 60%
Cross validation set: 20%
Test set: 20%
![](./images/model_selection_1.png)
![](./images/model_selection_2.png)

For Regularization:
![](./images/model_selection_3.png)

## Bias vs. Variance
![](./images/model_selection_4.png)
![](./images/model_selection_5.png)

### high bias
Low training set size: causes Jtrain(Θ) to be low and JCV(Θ) to be high.

Large training set size: causes both Jtrain(Θ) and JCV(Θ) to be high with Jtrain(Θ)≈JCV(Θ).

If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.

![](./images/model_selection_6.png)

### high variance

Low training set size: Jtrain(Θ) will be low and JCV(Θ) will be high.

Large training set size: Jtrain(Θ) increases with training set size and JCV(Θ) continues to decrease without leveling off. Also, Jtrain(Θ) < JCV(Θ) but the difference between them remains significant.

If a learning algorithm is suffering from high variance, getting more training data is likely to help.

![](./images/model_selection_7.png)

## kernel
![](./images/svm_1.png)
![](./images/svm_2.png)

# Unsupervised Learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like

## K-Means
![](./images/K-Means.png)

## PCA(Principal Component Analysis)
Compression:
Reduce memory/disk needed to store data
Speed up learning algorithm
Visualiza1on(2D/3D)

### covariance matrix
![](./images/cov_matrix.png)

## Anomaly detection
![](./images/anomaly_1.png)
![](./images/anomaly_2.png)
![](./images/anomaly_3.png)

## Recommender Systems
![](./images/recommender_1.png)
![](./images/recommender_2.png)

## Summary
![](./images/summary.png)




