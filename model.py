import numpy as np

class NeuralNetwork:
    def __init__(self, input_neurons, hidden_layers, output_neurons, init_wb_method, optimizer, activation, learning_rate, weight_decay, beta1, beta2, epsilon=1e-8):
        self.layers = [input_neurons] + hidden_layers + [output_neurons]
        self.weights = {}
        self.biases = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.init_wb_method = init_wb_method
        self.epsilon = epsilon
        self.activation = activation
        self.weight_decay = weight_decay

        if init_wb_method == 'random':
            for i in range(len(self.layers) - 1):
                self.weights[i] = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
                self.biases[i] = np.zeros((1, self.layers[i+1]))

        elif init_wb_method == 'xavier':
            for i in range(len(self.layers) - 1):
                r = np.sqrt(6.0 / (self.layers[i] + self.layers[i+1]))
                self.weights[i] = np.random.uniform(-r, r, (self.layers[i], self.layers[i+1]))
                self.biases[i] = np.zeros((1, self.layers[i+1]))

        # Initialize optimizer parameters
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        num_layers = len(self.layers) - 1  # Number of weight layers

        # For Adam, Nadam, and RMSprop
        self.m_w = {i: np.zeros_like(self.weights[i]) for i in range(num_layers)}
        self.v_w = {i: np.zeros_like(self.weights[i]) for i in range(num_layers)}
        self.m_b = {i: np.zeros_like(self.biases[i]) for i in range(num_layers)}
        self.v_b = {i: np.zeros_like(self.biases[i]) for i in range(num_layers)}
        
        # For Adam/Nadam timestep tracking
        self.t = 0

        # For Momentum and Nesterov
        self.velocity_w = {i: np.zeros_like(self.weights[i]) for i in range(num_layers)}
        self.velocity_b = {i: np.zeros_like(self.biases[i]) for i in range(num_layers)}


    # LIST OF ACTIVATION FUNCTIONS 

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Avoid overflow
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def softmax_derivative(self, output, y):
        return output - y
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, X):
        self.activations = {}
        self.z_values = {}

        # Input Layer
        self.activations[0] = X

        # Hidden Layers
        for i in range(len(self.layers) - 2):
            self.z_values[i] = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            if self.activation == 'relu':
                self.activations[i+1] = self.relu(self.z_values[i])
            elif self.activation == 'sigmoid':
                self.activations[i+1] = self.sigmoid(self.z_values[i])
            elif self.activation == 'tanh':
                self.activations[i+1] = self.tanh(self.z_values[i])

        # Output Layer (Softmax)
        last_layer = len(self.layers) - 2
        self.z_values[last_layer] = np.dot(self.activations[last_layer], self.weights[last_layer]) + self.biases[last_layer]
        self.activations[last_layer+1] = self.softmax(self.z_values[last_layer])

        return self.activations[last_layer+1]
    
    def backward(self, X, y):
        m = X.shape[0]

        # Compute gradient for the output layer
        output = self.activations[len(self.layers) - 1]
        delta = self.softmax_derivative(output, y) / m

        # Backpropagate through the layers
        gradients_w = {}
        gradients_b = {}

        for i in reversed(range(len(self.layers) - 1)):
            gradients_w[i] = np.dot(self.activations[i].T, delta)
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True)

            if i != 0:
                if self.activation == 'relu':
                    delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i - 1])
                elif self.activation == 'sigmoid':
                    delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.z_values[i - 1])
                elif self.activation == 'tanh':
                    delta = np.dot(delta, self.weights[i].T) * self.tanh_derivative(self.z_values[i - 1])

        return gradients_w, gradients_b
    
    def optimization_functions(self, gradients_w, gradients_b):
        num_layers = len(self.layers) - 1

        if self.optimizer == 'sgd':
            for i in range(num_layers):
                self.weights[i] -= self.learning_rate * (gradients_w[i] + self.weight_decay * self.weights[i])
                self.biases[i] -= self.learning_rate * gradients_b[i]  # No weight decay for biases

        elif self.optimizer == 'momentum':
            beta = 0.9
            for i in range(num_layers):
                self.velocity_w[i] = beta * self.velocity_w[i] + (1 - beta) * gradients_w[i]
                self.velocity_b[i] = beta * self.velocity_b[i] + (1 - beta) * gradients_b[i]

                self.weights[i] -= self.learning_rate * (self.velocity_w[i] + self.weight_decay * self.weights[i])
                self.biases[i] -= self.learning_rate * self.velocity_b[i]  # No weight decay for biases

        elif self.optimizer == 'nesterov':
            beta = 0.9
            for i in range(num_layers):
                prev_velocity_w = self.velocity_w[i]
                prev_velocity_b = self.velocity_b[i]

                self.velocity_w[i] = beta * prev_velocity_w + (1 - beta) * gradients_w[i]
                self.velocity_b[i] = beta * prev_velocity_b + (1 - beta) * gradients_b[i]

                self.weights[i] -= self.learning_rate * ((gradients_w[i] + beta * prev_velocity_w) + self.weight_decay * self.weights[i])
                self.biases[i] -= self.learning_rate * (gradients_b[i] + beta * prev_velocity_b)  # No weight decay for biases

        elif self.optimizer == 'rmsprop':
            decay_rate = 0.9
            epsilon = 1e-8
            for i in range(num_layers):
                self.m_w[i] = decay_rate * self.m_w[i] + (1 - decay_rate) * gradients_w[i]**2
                self.m_b[i] = decay_rate * self.m_b[i] + (1 - decay_rate) * gradients_b[i]**2

                self.weights[i] -= self.learning_rate * (gradients_w[i] / (np.sqrt(self.m_w[i]) + epsilon) + self.weight_decay * self.weights[i])
                self.biases[i] -= self.learning_rate * (gradients_b[i] / (np.sqrt(self.m_b[i]) + epsilon))  # No weight decay for biases

        elif self.optimizer == 'adam':
            self.t += 1
            for i in range(num_layers):
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients_w[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * gradients_w[i]**2
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients_b[i]
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * gradients_b[i]**2

                m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

                self.weights[i] -= self.learning_rate * (m_w_hat / (np.sqrt(v_w_hat) + self.epsilon) + self.weight_decay * self.weights[i])
                self.biases[i] -= self.learning_rate * (m_b_hat / (np.sqrt(v_b_hat) + self.epsilon))  # No weight decay for biases

        elif self.optimizer == 'nadam':
            self.t += 1
            for i in range(num_layers):
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients_w[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * gradients_w[i]**2
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients_b[i]
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * gradients_b[i]**2

                m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

                self.weights[i] -= self.learning_rate * ((self.beta1 * m_w_hat + (1 - self.beta1) * gradients_w[i]) / (np.sqrt(v_w_hat) + self.epsilon) + self.weight_decay * self.weights[i])
                self.biases[i] -= self.learning_rate * ((self.beta1 * m_b_hat + (1 - self.beta1) * gradients_b[i]) / (np.sqrt(v_b_hat) + self.epsilon))  # No weight decay for biases
                    
            
    def train(self, X, y, epochs, batch_size, X_val, y_val,loss_function = 'squared_error', wb_log=True,plots=False):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_loss = 0
            correct_preds_train = 0

            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward pass
                predictions = self.forward(X_batch)

                # Compute loss (Cross-entropy loss)
                
                if loss_function == "cross_entropy":
                    loss = -np.sum(y_batch * np.log(predictions + 1e-8)) / len(y_batch)  # Avoid log(0)
                elif loss_function == "squared_error":
                    loss = np.sum((y_batch - predictions) ** 2)
                total_loss += loss

                # Compute accuracy for training batch
                y_pred_labels = np.argmax(predictions, axis=1)
                y_true_labels = np.argmax(y_batch, axis=1)
                correct_preds_train += np.sum(y_pred_labels == y_true_labels)

                # Backward pass
                gradients_w, gradients_b = self.backward(X_batch, y_batch)

                # Update weights
                self.optimization_functions(gradients_w, gradients_b)

            # Compute training loss and accuracy
            avg_train_loss = total_loss / (len(X) / batch_size)
            train_losses.append(avg_train_loss)
            train_acc = correct_preds_train / len(X)
            train_accuracies.append(train_acc)

            # Compute validation loss and accuracy
            if X_val is not None and y_val is not None:
                val_predictions = self.forward(X_val)
                val_loss = -np.sum(y_val * np.log(val_predictions + 1e-8)) / len(y_val)
                val_losses.append(val_loss)

                # Compute validation accuracy
                val_pred_labels = np.argmax(val_predictions, axis=1)
                val_true_labels = np.argmax(y_val, axis=1)
                val_acc = np.sum(val_pred_labels == val_true_labels) / len(y_val)
                val_accuracies.append(val_acc)


                

                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")


        

        

        

        return train_acc, val_acc, avg_train_loss ,val_loss


            
    def predict(self, X_test):
        predictions = self.forward(X_test)
        # print(predictions)
        # y_pred_labels = np.argmax(predictions, axis=1)
        return predictions
        

# # Example Usage:

# hidden_layers = [128, 64]
    
# # Initialize model
# model = NeuralNetwork(input_neurons=784, hidden_layers=hidden_layers, output_neurons=10,init_wb_method='random',activation='tanh',learning_rate=0.001,optimizer='nadam',weight_decay=0.0005,beta1=0.9,beta2=0.99)

# # Train (Assume train_images is your training data and train_labels is one-hot encoded labels)
# model.train(train_images_splitted,train_labels_splitted,epochs=10,batch_size=32,X_val=val_images,y_val=val_labels,plots=True,wb_log=False)