import numpy as np
import matplotlib.pyplot as plt

# Function to plot the points and the decision boundary
def plot_perceptron(points, labels, weights):
    # Convertir les points et les poids en tableaux numpy
    points = np.array(points)
    labels = np.array(labels)
    
    # Extraire les points classés en 0 et 1
    class_0 = points[labels == 0]
    class_1 = points[labels == 1]

    # Tracer les points
    plt.scatter(class_0[:, 0], class_0[:, 1], color='red', label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class 1')
    
    # Calculer la frontière de décision (w0 + w1*x + w2*y = 0)
    x_values = np.array([min(points[:, 0]), max(points[:, 0])])
    y_values = -(weights[0] + weights[1] * x_values) / weights[2]
    plt.plot(x_values, y_values, 'g-', label='Decision Boundary')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Perceptron Decision Boundary')
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
    iterations = 0
    for epoch in range(epochs):
        all_correctly_classified = True
        for x, t in zip(points, labels):
            y = 1 if np.dot(weights.T, x) > 0 else 0
            if y != t:
                all_correctly_classified = False
                # Update weights incrementally after each example
                weights += alpha * (t - y) * x
                iterations += 1

        # If all points are correctly classified, break the loop
        if all_correctly_classified:
            break

    return weights.tolist(),iterations

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
        iterations = 0

        for x, t in zip(points, labels):
            # Calculate the perceptron output
            y = 1 if np.dot(weights.T, x) > 0 else 0
            
            # If the output is not equal to the desired label, update delta_w
            if y != t:
                all_correctly_classified = False
                delta_w += alpha * (t - y) * x
        
        # Update weights
        weights += delta_w
        iterations += 1        

        # If all points are correctly classified, break the loop
        if all_correctly_classified:
            break
    return weights.tolist(),iterations

def generate_ls_data_wide_range(N, dim, weights_teacher):
    """
    Generate linearly separable data points and their labels using a teacher perceptron.
    Fix: Generate points within a wider range [-100, 100] in the given dimension.
    
    :param N: Number of points to generate.
    :param dim: Dimension of the input space (excluding bias).
    :param weights_teacher: Weights of the teacher perceptron, including bias as the first element.
    :return: Tuple of points and labels.
    """
    # Generate random points within an even wider range [-100, 100] in the given dimension
    points = np.random.uniform(-100, 100, (N, dim))
    
    # Add a column of ones for the bias term
    points_with_bias = np.insert(points, 0, 1, axis=1)
    
    # Label the points based on the teacher perceptron
    labels = np.where(np.dot(points_with_bias, weights_teacher) > 0, 1, 0)
    
    return points, labels

N = 1000  # Number of points
dim = 3  # Dimension of each point (excluding bias)
weights_teacher = np.random.randn(dim + 1)

# Generate data with the wider range
points_T, labels_T = generate_ls_data_wide_range(N, dim, weights_teacher)
print("visualisation des points LS")
plot_perceptron(points_T, labels_T, weights_teacher)


# Poids initiaux pour les perceptrons élèves
initial_weights_student = np.random.randn(dim + 1)

# Taux d'apprentissage
alpha = 0.7

# Entraîner le perceptron élève en utilisant version batch
final_weights_batch,ite_batch = perceptron_batch(points_T, labels_T, initial_weights_student, alpha)
print("Batch Perceptron Final Weights:", final_weights_batch)
plot_perceptron(points_T, labels_T, final_weights_batch)
print(ite_batch)

# Entraîner le perceptron élève en utilisant version online
final_weights_online, ite_online = perceptron_online(points_T, labels_T, initial_weights_student, alpha)
print("Online Perceptron Final Weights:", final_weights_online)
plot_perceptron(points_T, labels_T, final_weights_online)
print(ite_online)

def calculate_overlap(weights_teacher, weights_student):
    """
    Calculate the overlap (R) between the teacher perceptron weights (W*) and the student perceptron weights (W).
    R = cos [ (W* · W) / (|W*| * |W|) ]
    
    :param weights_teacher: Weights of the teacher perceptron as a list [w0, w1, w2].
    :param weights_student: Weights of the student perceptron as a list [w0, w1, w2].
    :return: The overlap R between W* and W.
    """
    # Convert weights to numpy arrays
    weights_teacher = np.array(weights_teacher)
    weights_student = np.array(weights_student)
    
    # Calculate dot product (W* · W)
    dot_product = np.dot(weights_teacher, weights_student)
    
    # Calculate norms |W*| and |W|
    norm_teacher = np.linalg.norm(weights_teacher)
    norm_student = np.linalg.norm(weights_student)
    
    # Calculate the cosine of the angle between W* and W
    cosine_angle = dot_product / (norm_teacher * norm_student)
    
    return cosine_angle

# Calculate overlap for both batch and online student perceptrons
overlap_batch = calculate_overlap(weights_teacher, final_weights_batch)
overlap_online = calculate_overlap(weights_teacher, final_weights_online)

print(overlap_batch)
print(overlap_online)

