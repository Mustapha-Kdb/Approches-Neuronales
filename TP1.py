import numpy as np
import matplotlib.pyplot as plt

def plot_perceptron(points, labels, weights1,weights2):
    
    points = np.array(points)
    labels = np.array(labels)
    
    class_0 = points[labels == 0]
    class_1 = points[labels == 1]

    # Plot points
    plt.scatter(class_0[:, 0], class_0[:, 1], color='red', label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class 1')
    
    
    x_values = np.array([points[:, 0].min(), points[:, 0].max()])
    y_values = -(weights1[0] + weights1[1] * x_values) / weights1[2]
    
    x_values2 = np.array([points[:, 0].min(), points[:, 0].max()])
    y_values2 = -(weights2[0] + weights2[1] * x_values) / weights2[2]

    
    
    
    plt.plot(x_values, y_values, color='black', label='Batch')
    plt.plot(x_values2, y_values2, color='green', label='Online')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Perceptron')
    plt.legend()
    plt.grid(True)
    plt.show()

def perceptron_online(points, labels, weights, alpha, epochs=1000):
    points = np.array(points)
    labels = np.array(labels)
    weights = np.array(weights)
    points = np.insert(points, 0, 1, axis=1)  
    iterations = 0
    for epoch in range(epochs):
        all_correctly_classified = True
        # Mélanger les indices pour obtenir un ordre aléatoire à chaque époque
        indices = np.arange(len(points))
        np.random.shuffle(indices)
        for i in indices:
            x = points[i]
            t = labels[i]
            y = 1 if np.dot(weights.T, x) > 0 else 0
            if y != t:
                all_correctly_classified = False 
                weights += alpha * (t - y) * x
                iterations += 1
        if all_correctly_classified:
            break
    return weights.tolist()

    return weights.tolist()
def perceptron_batch(points, labels, weights, alpha):
   
    points = np.array(points)
    labels = np.array(labels)
    weights = np.array(weights)

    points = np.insert(points, 0, 1, axis=1)
    
    while True:
        delta_w = np.zeros_like(weights)
        all_correctly_classified = True
        
        for x, t in zip(points, labels):
            y = 1 if np.dot(weights.T, x) > 0 else 0
            
            if y != t:
                all_correctly_classified = False
                delta_w += alpha * (t - y) * x
        
        weights += delta_w
        if all_correctly_classified:
            break
    return weights.tolist()


points_example = [(0, 0), (0, 1), (1, 0), (1, 1)]
labels_example = [0, 0, 0, 1]
initial_weights = [0.5, 0.7, 0.3]  # w0, w1, w2
learning_rate = 0.45

final_weights1=perceptron_batch(points_example, labels_example, initial_weights, learning_rate)
print(final_weights1)

final_weights2=perceptron_online(points_example, labels_example, initial_weights,learning_rate)
print(final_weights2)
plot_perceptron( points_example, labels_example, final_weights1,final_weights2)


# "ou" exemple
points_example = [(0, 0), (0, 1), (1, 0), (1, 1)]
labels_example = [0, 1, 1, 1]

final_weights1=perceptron_batch(points_example, labels_example, initial_weights, learning_rate)
print(final_weights1)

final_weights2=perceptron_online(points_example, labels_example, initial_weights, learning_rate)
print(final_weights2)
plot_perceptron( points_example, labels_example, final_weights1,final_weights2)


points_cls = [(2, 1), (0, -1), (-2, 1), (0, 2)]
labels_cls = [1, 1, 0, 0]

final_weights1=perceptron_batch(points_cls, labels_cls, initial_weights, learning_rate)
print(final_weights1)

final_weights2=perceptron_online(points_cls, labels_cls, initial_weights, 0.75)
print(final_weights2)
plot_perceptron( points_cls, labels_cls, final_weights1,final_weights2)
