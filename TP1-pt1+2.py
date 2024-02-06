import numpy as np
import matplotlib.pyplot as plt


def plot_perceptron(points, labels, weights1,weights2):

    
    points = np.array(points)
    labels = np.array(labels)
    
    
    class_0 = points[labels == 0]
    class_1 = points[labels == 1]

    
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
        for x, t in zip(points, labels):
            y = 1 if np.dot(weights.T, x) > 0 else 0
            if y != t:
                all_correctly_classified = False
                
                weights += alpha * (t - y) * x
                iterations += 1

        
        if all_correctly_classified:
            break

    return weights.tolist(),iterations

def perceptron_batch(points, labels, weights, alpha):

    
    points = np.array(points)
    labels = np.array(labels)
    weights = np.array(weights)

    
    points = np.insert(points, 0, 1, axis=1)
    
    while True:
        delta_w = np.zeros_like(weights)
        all_correctly_classified = True
        iterations = 0

        for x, t in zip(points, labels):
            
            y = 1 if np.dot(weights.T, x) > 0 else 0
            
            
            if y != t:
                all_correctly_classified = False
                delta_w += alpha * (t - y) * x
        
        
        weights += delta_w
        iterations += 1        

        
        if all_correctly_classified:
            break
    return weights.tolist(),iterations

def generate_ls_data_wide_range(N, dim, weights_teacher):

    
    points = np.random.uniform(-100, 100, (N, dim))
    
    
    points_with_bias = np.insert(points, 0, 1, axis=1)
    
    
    labels = np.where(np.dot(points_with_bias, weights_teacher) > 0, 1, 0)
    
    return points, labels

N = 250  
dim = 2  
weights_teacher = np.random.randn(dim + 1)


points_T, labels_T = generate_ls_data_wide_range(N, dim, weights_teacher)
print("visualisation des points LS")
plot_perceptron(points_T, labels_T, weights_teacher,weights_teacher)



initial_weights_student = np.random.randn(dim + 1)


alpha = 0.7


final_weights_batch,ite_batch = perceptron_batch(points_T, labels_T, initial_weights_student, alpha)
print("Batch Perceptron Final Weights:", final_weights_batch)
print(ite_batch)


final_weights_online, ite_online = perceptron_online(points_T, labels_T, initial_weights_student, alpha)
print("Online Perceptron Final Weights:", final_weights_online)
plot_perceptron(points_T, labels_T, final_weights_batch,final_weights_online)
print(ite_online)

def calculate_overlap(weights_teacher, weights_student):
    
    weights_teacher = np.array(weights_teacher)
    weights_student = np.array(weights_student)
    
    
    dot_product = np.dot(weights_teacher, weights_student)
    
    
    norm_teacher = np.linalg.norm(weights_teacher)
    norm_student = np.linalg.norm(weights_student)
    
    
    cosine_angle = dot_product / (norm_teacher * norm_student)
    
    return cosine_angle


overlap_batch = calculate_overlap(weights_teacher, final_weights_batch)
overlap_online = calculate_overlap(weights_teacher, final_weights_online)

print(overlap_batch)
print(overlap_online)

