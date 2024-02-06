import numpy as np
import matplotlib.pyplot as plt

# Function to plot the points and the decision boundary
def plot_perceptron(points, labels, weights1,weights2):
    """
    Plot the two-dimensional points, their labels, and the perceptron decision boundary.
    
    :param points: List of two-dimensional points.
    :param labels: List of labels corresponding to the points.
    :param weights: Weights of the perceptron as a list [w0, w1, w2].
    """
    # Convert points and weights to numpy arrays
    points = np.array(points)
    labels = np.array(labels)
    
    # Extract points classified as 0 and 1
    class_0 = points[labels == 0]
    class_1 = points[labels == 1]

    # Plot points
    plt.scatter(class_0[:, 0], class_0[:, 1], color='red', label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class 1')
    
    # Calculate decision boundary (w0 + w1*x + w2*y = 0)
    # Define two points and plot the line between them (0, -w0/w2) and (-w0/w1, 0) pour la version batch
    x_values = np.array([points[:, 0].min(), points[:, 0].max()])
    y_values = -(weights1[0] + weights1[1] * x_values) / weights1[2]
    
    # Define two points and plot the line between them (0, -w0/w2) and (-w0/w1, 0) pour la version online
    x_values2 = np.array([points[:, 0].min(), points[:, 0].max()])
    y_values2 = -(weights2[0] + weights2[1] * x_values) / weights2[2]

    
    
    
    # Plot the two lines between the points
    plt.plot(x_values, y_values, color='black', label='Batch')
    plt.plot(x_values2, y_values2, color='green', label='Online')
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Perceptron')
    plt.legend()
    plt.grid(True)
    plt.show()

def perceptron_online(points, labels, weights, alpha, epochs=1000):
    """
    Implement a perceptron using the online update rule.
    
    :param points: List of two-dimensional points.
    :param labels: List of labels corresponding to the points.
    :param weights: Initial weights as a list [w0, w1, w2].
    :param alpha: Learning rate.
    :param epochs: Maximum number of epochs to run the algorithm.
    :return: Updated weights after running the perceptron algorithm.
    """
    points = np.array(points)
    labels = np.array(labels)
    weights = np.array(weights)
    points = np.insert(points, 0, 1, axis=1)  # Add bias term

    for epoch in range(epochs):
        all_correctly_classified = True
        for x, t in zip(points, labels):
            y = 1 if np.dot(weights.T, x) > 0 else 0
            if y != t:
                all_correctly_classified = False
                # Update weights incrementally after each example
                weights += alpha * (t - y) * x
        # If all points are correctly classified, break the loop
        if all_correctly_classified:
            break

    return weights.tolist()
def perceptron_batch(points, labels, weights, alpha):
    """
    Implement a perceptron using the batch update rule.
    
    :param points: List of two-dimensional points.
    :param labels: List of labels corresponding to the points.
    :param weights: Initial weights as a list [w0, w1, w2].
    :param alpha: Learning rate.
    :return: Updated weights after running the perceptron algorithm.
    """
    # Convert points and weights to numpy arrays for vectorized operations
    points = np.array(points)
    labels = np.array(labels)
    weights = np.array(weights)

    # Add a column of ones to the points to account for the bias (w0)
    points = np.insert(points, 0, 1, axis=1)
    
    while True:
        delta_w = np.zeros_like(weights)
        all_correctly_classified = True
        
        for x, t in zip(points, labels):
            # Calculate the perceptron output
            y = 1 if np.dot(weights.T, x) > 0 else 0
            
            # If the output is not equal to the desired label, update delta_w
            if y != t:
                all_correctly_classified = False
                delta_w += alpha * (t - y) * x
        
        # Update weights
        weights += delta_w
        # If all points are correctly classified, break the loop
        if all_correctly_classified:
            break
    return weights.tolist()

# Example usage:
# "et" exemple
# Define a set of points, labels, initial weights and learning rate
points_example = [(0, 0), (0, 1), (1, 0), (1, 1)]
labels_example = [0, 0, 0, 1]
initial_weights = [0.5, 0.7, 0.3]  # w0, w1, w2
learning_rate = 0.45

# Run the perceptron function version batch
final_weights1=perceptron_batch(points_example, labels_example, initial_weights, learning_rate)
print(final_weights1)

# Run the perceptron function version online
final_weights2=perceptron_online(points_example, labels_example, initial_weights,learning_rate)
print(final_weights2)
plot_perceptron( points_example, labels_example, final_weights1,final_weights2)


# "ou" exemple
# Define a set of points, labels, initial weights and learning rate
points_example = [(0, 0), (0, 1), (1, 0), (1, 1)]
labels_example = [0, 1, 1, 1]

# Run the perceptron function
final_weights1=perceptron_batch(points_example, labels_example, initial_weights, learning_rate)
print(final_weights1)

# Run the perceptron function version online
final_weights2=perceptron_online(points_example, labels_example, initial_weights, learning_rate)
print(final_weights2)
plot_perceptron( points_example, labels_example, final_weights1,final_weights2)


points_cls = [(2, 1), (0, -1), (-2, 1), (0, 2)]
labels_cls = [1, 1, 0, 0]

# Run the perceptron function version batch
final_weights1=perceptron_batch(points_cls, labels_cls, initial_weights, learning_rate)
print(final_weights1)

# Run the perceptron function version online
final_weights2=perceptron_online(points_cls, labels_cls, initial_weights, 0.75)
print(final_weights2)
# Use the initial weights for plotting since the weights were not updated
plot_perceptron( points_cls, labels_cls, final_weights1,final_weights2)
