import numpy as np
import matplotlib.pyplot as plt

def plot_perceptron(points, labels, weights):
    points = np.array(points)
    labels = np.array(labels)
    
    class_0 = points[labels == 0]
    class_1 = points[labels == 1]

    plt.scatter(class_0[:, 0], class_0[:, 1], color='red', label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class 1')
    
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
    return weights.tolist(), iterations

def perceptron_batch(points, labels, weights, alpha):
   
    points = np.array(points)
    labels = np.array(labels)
    weights = np.array(weights)
    iterations = 0
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
        iterations += 1        

        if all_correctly_classified:
            break
    return weights.tolist(),iterations

def generate_ls_data_wide_range(N, dim, weights_teacher):
   
    points = np.random.uniform(-5000, 5000, (N, dim))
    
    points_with_bias = np.insert(points, 0, 1, axis=1)
    
    labels = np.where(np.dot(points_with_bias, weights_teacher) > 0, 1, 0)
    
    return points, labels

def calculate_overlap(weights_teacher, weights_student):
    
    weights_teacher = np.array(weights_teacher)
    weights_student = np.array(weights_student)
    
    dot_product = np.dot(weights_teacher, weights_student)
    
    norm_teacher = np.linalg.norm(weights_teacher)
    norm_student = np.linalg.norm(weights_student)
    
    cosine_angle = dot_product / (norm_teacher * norm_student)
    
    return cosine_angle

def run_tests(N_values, P_values, eta_values, trials=50):
    results_batch = {}
    results_online = {}

    for N in N_values:
        for P in P_values:
            for eta in eta_values:
                print(f'Running tests for N={N}, P={P}, and eta={eta}...')
                overlap_batch = []
                overlap_online = []
                iterations_batch = []
                iterations_online = []

                for trial in range(trials):
                    weights_teacher = np.random.uniform(-1, 1, P + 1)  # P + 1 to account for the bias term

                    points, labels = generate_ls_data_wide_range(N, P, weights_teacher)

                    weights_student = np.random.uniform(-1, 1, P + 1)

                    weights_batch, it_batch = perceptron_batch(points, labels, weights_student, eta)
                    iterations_batch.append(it_batch)
                    overlap_batch.append(calculate_overlap(weights_teacher, weights_batch))
                    
                    weights_online, it_online = perceptron_online(points, labels, weights_student, eta)
                    iterations_online.append(it_online)
                    overlap_online.append(calculate_overlap(weights_teacher, weights_online))

                results_batch[(N, P, eta)] = (np.mean(overlap_batch), np.mean(iterations_batch))
                results_online[(N, P, eta)] = (np.mean(overlap_online), np.mean(iterations_online))
    return results_batch, results_online

def format_results(N_values, P_values, eta_values, results_batch, results_online):
    for eta in eta_values:
        print(f'eta={eta}')
        print('+--------+' + '+------------------' * len(P_values) + '+')
        print('|   N\P   | ' + ' | '.join(f'{P:>16}' for P in P_values) + ' |')
        print('+--------+' + '+------------------' * len(P_values) + '+')
        for N in N_values:
            batch_results = ' | '.join(f'<{results_batch[(N, P, eta)][0]:.2f};{results_batch[(N, P, eta)][1]:.2f}>' for P in P_values)
            online_results = ' | '.join(f'<{results_online[(N, P, eta)][0]:.2f};{results_online[(N, P, eta)][1]:.2f}>' for P in P_values)
            print(f'| {N:<6} | {batch_results} |')
            print('+--------+' + '+------------------' * len(P_values) + '+')
            print(f'| {N:<6} | {online_results} |')
            print('+--------+' + '+------------------' * len(P_values) + '+')
        print('\n')

    
# Configurations des tests
N_values = [2, 10, 100, 500, 1000]
P_values = [10, 100, 500, 1000]
eta_values = [0.45, 0.45/2, 0.45/10]

results_batch, results_online = run_tests(N_values, P_values, eta_values)

format_results(N_values, P_values, eta_values, results_batch, results_online)
