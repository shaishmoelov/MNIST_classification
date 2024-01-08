# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Fetch MNIST dataset
mnist = datasets.fetch_openml('mnist_784', version=1)

# Extract features and labels
X, y = mnist.data, mnist.target.astype(int)

# Data partitioning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# Transform labels to one-hot vectors
def one_hot_encode(labels, num_classes=10):
    encoded_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded_labels[i, label] = 1
    return encoded_labels

y_train_one_hot = one_hot_encode(y_train)
y_test_one_hot = one_hot_encode(y_test)

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def train(self, X, y):
        # Initialize weights
        self.w = np.zeros((X.shape[1], len(np.unique(y))))
        self.errors = []

        for _ in range(self.max_iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - np.dot(xi, self.w))
                self.w[:, target] += update * xi
                errors += int(update != 0.0)
            self.errors.append(errors)

            if errors == 0:
                break

    def predict(self, X):
        return np.argmax(np.dot(X, self.w), axis=1)


# Initialize Perceptron
perceptron = Perceptron(learning_rate=0.01, max_iterations=1000)

# Train models for each class
for i in range(10):
    binary_labels = np.where(y_train == i, 1, -1)
    perceptron.train(X_train, binary_labels)

# Make predictions on test data
y_pred = perceptron.predict(X_test)

# Initialize Perceptron
perceptron = Perceptron(learning_rate=0.01, max_iterations=1000)

# Train models for each class
for i in range(10):
    binary_labels = np.where(y_train == i, 1, -1)
    perceptron.train(X_train, binary_labels)

# Make predictions on test data
y_pred = perceptron.predict(X_test)