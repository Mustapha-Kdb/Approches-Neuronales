import numpy as np
import matplotlib.pyplot as plt

# Function to plot the points and the decision boundary
def plot_perceptron(points, labels, weights):
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
    # Define two points and plot the line between them (0, -w0/w2) and (-w0/w1, 0)
    point1 = [0, -weights[0] / weights[2]]
    point2 = [-weights[0] / weights[1], 0]

    print (point1, point2)
    
    # Plot the line between the two points
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], label='Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Perceptron Decision Boundary')
    plt.grid(True)
    plt.show()
def perceptron_online(points, labels, weights, alpha):
    """
    Implement a perceptron using the online update rule.
    
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
        
        # for x, t in zip(points, labels):
        # choisie une paire aléatoire de x et t 
        for i in range(len(points)):
            t= labels[i]
            x= points[i]
            # Calculate the perceptron output
            y = 1 if np.dot(weights.T, x) > 0 else 0
            
            # If the output is not equal to the desired label, update delta_w
            if y != t:
                all_correctly_classified = False
                delta_w = alpha * (t - y) * x
        
        # Update weights
        weights += delta_w
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
learning_rate = 0.75

# Run the perceptron function version batch
final_weights=perceptron_batch(points_example, labels_example, initial_weights, learning_rate)
print(final_weights)
plot_perceptron( points_example, labels_example, final_weights)

# Run the perceptron function version online
final_weights=perceptron_online(points_example, labels_example, initial_weights, 0.99)
print(final_weights)
plot_perceptron( points_example, labels_example, final_weights)


# "ou" exemple
# Define a set of points, labels, initial weights and learning rate
points_example = [(0, 0), (0, 1), (1, 0), (1, 1)]
labels_example = [0, 1, 1, 1]
initial_weights = [0.5, 0.7, 0.3]  # w0, w1, w2
learning_rate = 0.75

# Run the perceptron function
final_weights=perceptron_batch(points_example, labels_example, initial_weights, learning_rate)
print(final_weights)
plot_perceptron( points_example, labels_example, final_weights)

# Run the perceptron function version online
final_weights=perceptron_online(points_example, labels_example, initial_weights, learning_rate)
print(final_weights)
plot_perceptron( points_example, labels_example, final_weights)


points_cls = [(2, 1), (0, -1), (-2, 1), (0, 2)]
labels_cls = [1, 1, 0, 0]
initial_weights = [0.5, -0.7, 0.2]  # w0, w1, w2
learning_rate = 0.1

# Run the perceptron function version batch
final_weights=perceptron_batch(points_cls, labels_cls, initial_weights, learning_rate)
print(final_weights)
# Use the initial weights for plotting since the weights were not updated
plot_perceptron(points_cls, labels_cls, final_weights)

# Run the perceptron function version online
final_weights=perceptron_online(points_cls, labels_cls, initial_weights, 0.75)
print(final_weights)
# Use the initial weights for plotting since the weights were not updated
plot_perceptron(points_cls, labels_cls, final_weights)


