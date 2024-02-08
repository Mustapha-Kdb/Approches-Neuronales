import pandas as pd
import numpy as np
from urllib.request import urlopen

def perceptron_online(X, y, epochs=1000, alpha=0.01):
    weights = np.zeros(X.shape[1] + 1)  # +1 for the bias
    for epoch in range(epochs):
        for i in range(len(X)):
            xi = np.insert(X[i], 0, 1)  # Insert bias term
            prediction = np.dot(weights, xi)
            if (prediction >= 0 and y[i] == 0) or (prediction < 0 and y[i] == 1):
                weights += alpha * (y[i] - (prediction >= 0)) * xi
    return weights

def predict(X, weights):
    X = np.insert(X, 0, 1, axis=1)  # Insert bias term to each data point
    predictions = np.dot(X, weights) >= 0
    return predictions

# URL du jeu de données du Sonar sur le site de l'UCI
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'

# Télécharger les données dans un DataFrame
data = pd.read_csv(urlopen(url), header=None, sep=',')

# Séparer les données marquées avec une étoile pour l'ensemble 'train'
train_data = data[data[60].str.contains('\*')]

# Séparer les données restantes pour l'ensemble 'test'
test_data = data[~data[60].str.contains('\*')]

# Enlever l'étoile des labels dans l'ensemble 'train'
train_data[60] = train_data[60].str.replace('\*', '', regex=True).str.strip()

# Convertir les DataFrame en format d'écho par ligne avec des tabs
train_data.to_csv('train_data.tsv', sep='\t', index=False, header=False)
test_data.to_csv('test_data.tsv', sep='\t', index=False, header=False)
data.to_csv('complete_data.tsv', sep='\t', index=False, header=False)

# Assuming train_data and test_data are already defined and preprocessed
X_train = train_data.iloc[:, :-1].values  # all but last column
y_train = train_data.iloc[:, -1].apply(lambda x: 1 if x == 'M' else 0).values  # last column, converting classes to 0/1

X_test = test_data.iloc[:, :-1].values  # all but last column, test data doesn't have labels

# Train the perceptron
weights = perceptron_online(X_train, y_train)

# Predict on training and test data
train_predictions = predict(X_train, weights)
test_predictions = predict(X_test, weights)

# Calculate errors
Ea = np.mean(train_predictions != y_train)
if np.isnan(Ea):
    Ea = 0  # Assign a default value if Ea is NaN

# Display errors
print("Learning error (Ea):", Ea)

# Display weights
print("Weights:", weights)
