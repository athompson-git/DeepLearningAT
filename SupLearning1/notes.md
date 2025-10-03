# Notes on Supervised Machine Learning: Regression and Classification

## Week 1

* supervised learning: trains on labeled data, more widely used. Maps x -> y outcomes.
* unsupervised learning: less widely used, looks for patterns in unlabeled data

Supervised learning
* regression: predict infinite possible outputs
* classification: discrete, finite set out possible outputs

Unsupervised learning
* Clustering unlabeled N-dimensional data into groups
* Anomaly detection
* dimensionality reduction


### Supervised learning
$x \to [f] \to y$

We map inputs $x^i$ to outputs (target variable) $y^i$ for training example $i$ with $m$ training examples.

Cost function example: squared error cost function

$J(w,b) = \frac{1}{2 m} \sum_{i=1}^m (\hat{y}_i - y_i)^2$

with the prediction $\hat{y}^i = f(w,b)^i$.

#### Gradient descent

want the minimum of the cost function $\min_{w,b} J(w,b)$

* start with initial guess for w,b (e.g. both zero)
* sometimes cost functions have very nontrivial (e.g. multimodal) shapes

we use the update rule
$$ w \to w - \alpha \frac{\partial}{\partial w} J(w,b) $$
$$ b \to b - \alpha \frac{\partial}{\partial b} J(w,b) $$
moving with the gradient of the cost function with respect to w and b as we move in the the space. This update takes place __simultaneously__. Here $\alpha$ is the __learning rate__.

E.g.
```
tmp_w = w - alpha * grad_w
tmp_b = b - alpha * grad_b

w = tmp_w
b = tmp_b
```


## Week 2


## Week 3

