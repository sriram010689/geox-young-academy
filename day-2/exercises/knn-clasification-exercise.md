KNN - Classification Exercise
===============

Goals
-----

- Get familiar with the classifier interface of scikit learn
- Understand the importance of separate train and test datasets

Requirements
-----
- numpy
- scikit learn
- matplotlib

Exercise
-----

Consider the following code:
```python
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.random.seed(0)

world_is_nice = True

def drawSamples(clusterSize):
	variance = 0.3
	if (not world_is_nice):
		variance = 0.75

	X = np.random.normal([-1.0, -1.0], [variance, variance], size=[clusterSize, 2])
	Y = np.full(shape=clusterSize, fill_value=0)

	X = np.concatenate((X, np.random.normal([-1.0, 1.0], [variance, variance], size=[clusterSize, 2])))
	Y = np.concatenate((Y, np.full(shape=clusterSize, fill_value=1)))

	X = np.concatenate((X, np.random.normal([1.0, -1.0], [variance, variance], size=[clusterSize, 2])))
	Y = np.concatenate((Y, np.full(shape=clusterSize, fill_value=1)))

	X = np.concatenate((X, np.random.normal([1.0, 1.0], [variance, variance], size=[clusterSize, 2])))
	Y = np.concatenate((Y, np.full(shape=clusterSize, fill_value=0)))
	return X, Y
```

Instead of using actual, real data, which is hard to come by, we generate data for a simple toy example. The function drawSamples() generates these data samples with corresponding labels which we want to use for training (and testing). When called like this:
```
(X, Y) = drawSamples(100)
```
X is a matrix that contains the 2D features and Y is an array that contains the labels (either 0 or 1).

### Task 1

Write a function plotCircles(X, Y) which plots the data as a scatter plot of red and blue circles (depending on the label). 
Hint:
```
# This plots red circles
plt.plot(x_positions, y_positions, 'ro')
# And this plots blue circles
plt.plot(x_positions, y_positions, 'bo')
```
Use your function to look at the data distribution.


### Task 2

"Train" a k-nearest neighbor classifier on a training dataset of 4x100 samples. Training a scikit learn classifier involves two steps:
1. Initialization
2. Training/Fitting

```
#1
clf = neighbors.KNeighborsClassifier(num_neighbors, weights='uniform')
#2
clf.fit(data, labels)
```

Use the following function to display the learned decision boundaries:
```
def plotDecisionBoundaries(train_X, classifier, resolution):
	x_min, x_max = train_X[:, 0].min() - 1, train_X[:, 0].max() + 1
	y_min, y_max = train_X[:, 1].min() - 1, train_X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
		                 np.arange(y_min, y_max, resolution))
	Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]

	Z = Z.reshape(xx.shape)
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu)
```
To be called like this:
```
(X, Y) = drawSamples(100)
# ... train classifier called clf
plotDecisionBoundaries(X, clf, 0.01)
```
Plot the training samples on top of that.


### Task 3

How good is the classifier? How well can it predict the label based on the 2D features? You could count (bad idea), or you could write a function that counts for you. Write a function testAccuracy(classifier, data, ground_truth_labels) that runs the classifier on the given data and compares the predicted labels to the given ground truth labels. The result should be the fraction of correctly predicted labels.

### Task 4

Switch the difficuly of the data by setting the boolean "world_is_nice" to False.

With which hyper parameter setting (number of neighbors) does the classifier perform best on the training data? What is the accuracy, and is that realistic?

In reality, we would not apply the classifier to our training data, but to new data. We want to use it after all.
What happens when you sample new data, lets call it "test data", and evaluate on this yet unseen data? What happens when you change the hyper parameter setting (number of neighbors) to the accuracy on training and testing datasets?








