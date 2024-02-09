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

    
    points = np.random.uniform(-100, 100, (N, dim))
    
    
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

def run_tests(dimensions, points, etas, perceptron_function):
    results = np.zeros((len(dimensions), len(points), len(etas), 2))  # 2 pour <IT> et <R>
    
    for i, N in enumerate(dimensions):
        for j, P in enumerate(points):
            for k, eta in enumerate(etas):
                iterations = []
                overlaps = []
                for _ in range(50):  # 50 tirages aléatoires
                    # Générer des données linéairement séparables
                    weights_teacher = np.random.randn(N + 1)
                    points_T, labels_T = generate_ls_data_wide_range(P, N, weights_teacher)
                    initial_weights_student = np.random.randn(N + 1)

                    # Entraînement du perceptron
                    final_weights, it = perceptron_function(points_T, labels_T, initial_weights_student, eta)
                    overlap = calculate_overlap(weights_teacher, final_weights)
                    
                    iterations.append(it)
                    overlaps.append(overlap)
                
                # Calcul de la moyenne pour les itérations et le rapport
                results[i, j, k, 0] = np.mean(iterations)
                results[i, j, k, 1] = np.mean(overlaps)
    
    return results

# Dimensions et nombres de points à tester
N_values = [2, 10, 100, 500, 1000, 5000]
P_values = [10, 100, 500,1000]
eta_values = [0.7, 0.7/2, 0.7/10]

# Exécuter les tests pour la version batch
results_batch = run_tests(N_values, P_values, eta_values, perceptron_batch)

# Exécuter les tests pour la version online
results_online = run_tests(N_values, P_values, eta_values, perceptron_online)

# Fonction pour afficher les résultats sous forme de tableau
def format_results(N_values, P_values, eta_values, results_batch, results_online):
    for eta_index, eta in enumerate(eta_values):
        print(f'eta={eta}')
        print('+--------+' + '+------------------' * len(P_values) + '+')
        print('|   N\P   | ' + ' | '.join(f'{P:>16}' for P in P_values) + ' |')
        print('+--------+' + '+------------------' * len(P_values) + '+')
        for N_index, N in enumerate(N_values):
            batch_results = ' | '.join(
                f'<{results_batch[N_index, j, eta_index, 0]:.2f};{results_batch[N_index, j, eta_index, 1]:.2f}>'
                for j, P in enumerate(P_values)
            )
            online_results = ' | '.join(
                f'<{results_online[N_index, j, eta_index, 0]:.2f};{results_online[N_index, j, eta_index, 1]:.2f}>'
                for j, P in enumerate(P_values)
            )
            print(f'| {N:<6} | {batch_results} |')
            print('+--------+' + '+------------------' * len(P_values) + '+')
            print(f'| {N:<6} | {online_results} |')
            print('+--------+' + '+------------------' * len(P_values) + '+')
        print('\n')


format_results(N_values, P_values, eta_values, results_batch, results_online)
