import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        '''
        Initialize Multilayer Perceptron (MLP) model.

        Parameters:
        - input_size (int): Number of input features.
        - hidden_size (int): Number of neurons in the hidden layer.
        - output_size (int): Number of output neurons.

        Initializes weights and biases with random values.
        '''
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
        '''
        Perform forward pass through the network and Returns the output of the network after forward pass.

        The forward pass computes the output of the neural network given the input data X.
        It involves the following steps:
        1. Compute the activation of the hidden layer neurons.
        2. Apply the activation function to the hidden layer activation.
        3. Compute the activation of the output layer neurons.
        4. Apply the activation function to the output layer activation.
        5. Return the output of the network.
        '''
        # Forward pass through the network
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden # Compute the activation of the hidden layer
        self.hidden_output = self.activation_function(self.hidden_activation)# Apply activation function to the hidden layer activation

        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output# Compute the activation of the output layer
        self.output = self.activation_function(self.output_activation)# Apply activation function to the output layer activation

        return self.output

    def activation_function(self, x):
        # Activation function (sigmoid in this case)
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        # Derivative of the activation function
        return x * (1 - x)

    def backward(self, X, y, learning_rate):
        
        # Compute the error between the predicted output and the actual output
        output_error = y - self.output
        # Compute the delta (gradient) for the output layer using the error and the derivative of the activation function
        output_delta = output_error * self.activation_derivative(self.output).reshape(-1, 1)

        # Compute the error for the hidden layer by backpropagating the error from the output layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        # Compute the delta (gradient) for the hidden layer using the error and the derivative of the activation function
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_output)

        # Update the weights between the hidden and output layers based on the hidden layer's output and the output delta
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        # Update the weights between the input and hidden layers based on the input data and the hidden delta
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate

        # Update the biases for the output layer based on the output delta and hidden delta
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

        return output_delta


    def train(self, X_train, y_train, X_val, y_val, epochs, initial_learning_rate, final_learning_rate):
        # Define weight decay
        weight_decay = 0.01  # You can adjust this value as needed
        losses_train = []  # List to store training loss values over epochs
        losses_val = []  # List to store validation loss values over epochs

        learning_rate = initial_learning_rate  # Initialize learning rate

        for epoch in range(epochs):
            output_train = self.forward(X_train)
            output_delta = self.backward(X_train, y_train, learning_rate)  # Get output_delta from backward pass
            loss_train = np.mean(np.square(y_train - output_train))
            losses_train.append(loss_train)

            # Compute validation loss
            output_val = self.forward(X_val)
            loss_val = np.mean(np.square(y_val - output_val))
            losses_val.append(loss_val)


            # Bold Driver
            error_change_threshold = 1.04  # Predefined error change threshold
            if epoch % 1000 == 0:  # Update every thousand epochs
                prev_loss = losses_train[-2] if len(losses_train) >= 2 else float('inf')  # Previous loss
                current_loss = losses_train[-1]  # Current loss

                if current_loss > prev_loss * error_change_threshold:  # Error increased
                    # Undo weight changes
                    self.weights_hidden_output += (np.dot(self.hidden_output.T, output_delta) - weight_decay * self.weights_hidden_output) * learning_rate
                    self.weights_input_hidden += (np.dot(X_train.T, hidden_delta) - weight_decay * self.weights_input_hidden) * learning_rate

                    # Decrease learning rate
                    learning_rate *= 0.7  # Decrease learning rate by 30%
                else:  # Error decreased or remained same
                    # Adjust learning rate
                    learning_rate *= 1.05  # Increase learning rate by 5%
                    learning_rate = min(max(learning_rate, 0.01), 0.75)  # Bound learning rate

            # Print loss (optional)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Training Loss: {loss_train}, Validation Loss: {loss_val}, Learning Rate: {learning_rate}')

        # Plot training loss
        plt.plot(range(len(losses_train)), losses_train, label='Training Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot validation loss
        plt.plot(range(len(losses_val)), losses_val, label='Testing Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Testing Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Calculate predicted output for the training data
        predicted_output_train = self.forward(X_train)

        # Compute squared differences between actual and predicted outputs for training data
        squared_errors_train = np.square(y_train - predicted_output_train)

        # Calculate Mean Squared Error (MSE) for training data
        mse_train = np.mean(squared_errors_train)

        print("Mean Squared Error (MSE) on Training Data:", mse_train)

        return mse_train


    def test(self, X_test, y_test):
        # Run forward pass on the test data
        predicted_output_test = self.forward(X_test)

        # Compute squared differences between actual and predicted outputs for test data
        squared_errors_test = np.square(y_test - predicted_output_test)

        # Calculate Mean Squared Error (MSE) for test data
        mse_test = np.mean(squared_errors_test)

        print("Mean Squared Error (MSE) on Test Data:", mse_test)

        # Optionally, visualize the predictions compared to the actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predicted_output_test, color='blue', label='Actual vs. Predicted')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfectly Predicted')
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title('Actual vs. Predicted Values on Test Data')
        plt.legend()
        plt.grid(True)
        plt.show()

        return mse_test


# Load data from CSV file
data = pd.read_csv("TrainData.csv", usecols=["AREA", "LDP", "Index flood"])
# Separate features (X_train) and target variable (y_train)
X_train = data[["AREA", "LDP"]].values
y_train = data["Index flood"].values.reshape(-1, 1)

val_data = pd.read_csv("ValidationData.csv", usecols=["AREA", "LDP", "Index flood"])
X_val = val_data[["AREA", "LDP"]].values
y_val = val_data["Index flood"].values.reshape(-1, 1)

# Load test data from CSV file
test_data = pd.read_csv("TestData.csv", usecols=["AREA", "LDP", "Index flood"])
X_test = test_data[["AREA", "LDP"]].values
y_test = test_data["Index flood"].values.reshape(-1, 1)

# Define the architecture of the MLP
input_size = 2
hidden_size = 4
output_size = 1
initial_learning_rate = 0.15
final_learning_rate = 0.01
epochs = 15000

# Create MLP object
mlp = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# Train the MLP
mse_train = mlp.train(X_train, y_train, X_val, y_val, epochs=epochs, initial_learning_rate=initial_learning_rate, final_learning_rate=final_learning_rate)

# Test the MLP on test data
mse_test = mlp.test(X_test, y_test)

# Predict output for the training data
predicted_output_train = mlp.forward(X_train)

# Plot actual vs. predicted values for training data
plt.figure(figsize=(10, 6))
plt.scatter(y_train, predicted_output_train, color='blue', label='Actual vs. Predicted (Training)')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--', label='Perfectly Predicted')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs. Predicted Values (Training)')
plt.legend()
plt.grid(True)
plt.show()
