import numpy as np

class QuantumNeuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.phase = np.random.uniform(0, 2*np.pi)

    def quantum_activation(self, x):
        # Quantum-inspired activation function
        return np.cos(x + self.phase) ** 2

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.quantum_activation(z)

class QuantumLayer:
    def __init__(self, num_inputs, num_neurons):
        self.neurons = [QuantumNeuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

class QuantumNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(QuantumLayer(layer_sizes[i-1], layer_sizes[i]))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, X, y, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            total_loss = 0
            for x, target in zip(X, y):
                # Forward pass
                output = self.forward(x)
                
                # Compute loss (using mean squared error)
                loss = np.mean((output - target) ** 2)
                total_loss += loss

                # Backward pass (simplified, not true backpropagation)
                for layer in reversed(self.layers):
                    for neuron in layer.neurons:
                        # Update weights and bias
                        neuron.weights -= learning_rate * 2 * (output - target) * x
                        neuron.bias -= learning_rate * 2 * (output - target)
                        
                        # Update phase
                        neuron.phase -= learning_rate * np.sin(2 * (np.dot(x, neuron.weights) + neuron.bias + neuron.phase))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")

# Example usage
if __name__ == "__main__":
    # XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    qnn = QuantumNeuralNetwork([2, 4, 1])
    qnn.train(X, y, learning_rate=0.1, epochs=1000)

    # Test the network
    for x in X:
        print(f"Input: {x}, Output: {qnn.forward(x)}")
