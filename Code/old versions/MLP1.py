import numpy as np
import pandas as pd

# Load data from CSV file
data = pd.read_csv("TrainData.csv", usecols=["AREA", "LDP", "Index flood"])

# Separate features (X_train) and target variable (y_train)
X_train = data[["AREA", "LDP"]].values
y_train = data["Index flood"].values.reshape(-1, 1)

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(0)
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    biases_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    biases_output = np.zeros((1, output_size))
    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

# Forward pass
def forward(X, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output):
    hidden_input = np.dot(X, weights_input_hidden) + biases_hidden
    hidden_output = sigmoid(hidden_input)
    output = np.dot(hidden_output, weights_hidden_output) + biases_output
    return output, hidden_output

# Adaline learning algorithm
def adaline_learning(X_train, y_train, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output, learning_rate):
    for i in range(len(X_train)):
        output, hidden_output = forward(X_train[i].reshape(1, -1), weights_input_hidden, biases_hidden, weights_hidden_output, biases_output)
        prediction = output[0][0]
        error = y_train[i] - prediction
        weights_hidden_output += learning_rate * error * hidden_output.T
        biases_output += learning_rate * error
        weights_input_hidden += learning_rate * error * X_train[i].reshape(-1, 1).dot(weights_hidden_output.T)
        biases_hidden += learning_rate * error * weights_hidden_output.T
    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

# Training loop
def train(X_train, y_train, input_size, hidden_size, output_size, learning_rate, epochs):
    # Initialize weights and biases
    weights_input_hidden, biases_hidden, weights_hidden_output, biases_output = initialize_parameters(input_size, hidden_size, output_size)
    
    # Training loop
    for epoch in range(epochs):
        # Adaline learning algorithm (replace perceptron_learning with adaline_learning)
        weights_input_hidden, biases_hidden, weights_hidden_output, biases_output = adaline_learning(X_train, y_train, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output, learning_rate)
        
        # Compute and print loss (optional)
        output, _ = forward(X_train, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output)
        loss = np.mean(np.square(y_train - output))
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

# Example usage
if __name__ == "__main__":
    # Parameters
    input_size = 2
    hidden_size = 3
    output_size = 1
    learning_rate = 0.05
    epochs = 1000

    # Train the model
    train(X_train, y_train, input_size, hidden_size, output_size, learning_rate, epochs)
