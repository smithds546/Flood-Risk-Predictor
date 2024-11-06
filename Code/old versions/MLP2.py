import numpy as np
import pandas as pd

# Load data from CSV file
data = pd.read_csv("TrainData.csv", usecols=["AREA", "LDP", "Index flood"])

# Separate features (X_train) and target variable (y_train)
X_train = data[["AREA", "LDP"]].values
y_train = data["Index flood"].values.reshape(-1, 1)

# Initialize weights
def initialize_weights(input_size):
    np.random.seed(0)
    return np.random.randn(input_size)

def adaline_learning(X_train, y_train, weights, learning_rate, epochs):
    for epoch in range(epochs):
        for i in range(len(X_train)):
            prediction = np.dot(X_train[i], weights)
            error = y_train[i] - prediction
            weights += learning_rate * error * X_train[i]
    return weights

# Training loop
def train(X_train, y_train, input_size, hidden_size, output_size, learning_rate, epochs):
    # Initialize weights
    weights = initialize_weights(input_size)
    
    # Training loop
    for epoch in range(epochs):
        # Adaline learning algorithm
        weights = adaline_learning(X_train, y_train, weights, learning_rate, epochs)
        
        # Compute and print loss (optional)
        output = np.dot(X_train, weights)
        loss = np.mean(np.square(y_train - output))
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

# Example usage
if __name__ == "__main__":
    # Parameters
    input_size = 2
    hidden_size = 6
    output_size = 1
    learning_rate = 0.05
    epochs = 1000

    # Train the model
    train(X_train, y_train, input_size, hidden_size, output_size, learning_rate, epochs)
