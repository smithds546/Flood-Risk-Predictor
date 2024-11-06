import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with random values
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

        # Initialize biases with random values
        self.bias_hidden = np.random.rand(1, self.hidden_size)
        self.bias_output = np.random.rand(1, self.output_size)

    def forward(self, X):
        # Forward pass through the network
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.activation_function(self.hidden_activation)

        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.activation_function(self.output_activation)

        return self.output

    def activation_function(self, x):
        # Activation function (sigmoid in this case)
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        # Derivative of the activation function
        return x * (1 - x)

    def backward(self, X, y, learning_rate):
        # Backpropagation
        output_error = y - self.output
        output_delta = output_error * self.activation_derivative(self.output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_output)

        # Adjusting weights and biases
        self.weights_hidden_output += (np.dot(self.hidden_output.T, output_delta)) * learning_rate
        self.weights_input_hidden += (np.dot(X.T, hidden_delta)) * learning_rate

        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    def train(self, X, y, epochs, initial_learning_rate, final_learning_rate):
        losses = []  # List to store the loss values over epochs
        for epoch in range(1, epochs + 1):
            output = self.forward(X)
            current_learning_rate = self.bold_driver(initial_learning_rate, final_learning_rate, epoch)
            self.backward(X, y, current_learning_rate)
            loss = np.mean(np.square(y - output))  # Compute the mean squared error
            losses.append(loss)  # Append the loss value to the list

            # Print loss (optional)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}, Learning Rate: {current_learning_rate}')

        # Plot the loss over epochs
        plt.plot(range(1, epochs + 1), losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.ylim(0, 0.1)  # Set y-axis limit from 0 to 0.1
        plt.yticks(np.arange(0, 0.11, step=0.01))  # Set y-axis ticks from 0 to 0.1
        plt.show()

##    def bold_driver(self, initial_learning_rate, final_learning_rate, epoch):
##        if epoch % 1000 == 0:  # Update learning rate every thousand epochs
##            if initial_learning_rate < final_learning_rate:
##                return initial_learning_rate * 1.1  # Increase learning rate by 10%
##            else:
##                return initial_learning_rate * 0.5  # Decrease learning rate by 50%
##        else:
##            return initial_learning_rate


# Load data from CSV file
data = pd.read_csv("TrainData.csv", usecols=["AREA", "LDP", "Index flood"])

# Separate features (X_train) and target variable (y_train)
X_train = data[["AREA", "LDP"]].values
y_train = data["Index flood"].values.reshape(-1, 1)

# Define the architecture of the MLP
input_size = 2
hidden_size = 8
output_size = 1
initial_learning_rate = 0.1
final_learning_rate = 0.01
epochs = 5000

# Create MLP object
mlp = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# Train the MLP
mlp.train(X_train, y_train, epochs=epochs, initial_learning_rate=initial_learning_rate, final_learning_rate=final_learning_rate)

# Predict output for the training data
predicted_output = mlp.forward(X_train)

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, predicted_output, color='blue', label='Actual vs. Predicted')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--', label='Perfectly Predicted')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.grid(True)
plt.show()
