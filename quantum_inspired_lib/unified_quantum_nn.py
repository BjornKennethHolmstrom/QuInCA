import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import expm
from typing import List, Tuple
import logging
from logger import setup_logger

class UnifiedQuantumNeuron:
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self.weights = unitary_group.rvs(self.num_inputs)  # Generate unitary matrix of size num_inputs
        self.bias = np.random.uniform(0, 2*np.pi)
        self.phase = np.random.uniform(0, 2*np.pi)

    def quantum_activation(self, x: np.ndarray) -> np.ndarray:
        # Apply a quantum-like activation function with phase
        return np.abs(np.cos(x + self.phase))**2 + 1j * np.abs(np.sin(x + self.phase))**2

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Debugging information
        print(f"Original input shape: {inputs.shape}")

        # Reshape inputs into 2D array (batch_size, num_inputs) if necessary
        if inputs.ndim > 2:
            flattened_inputs = inputs.reshape(inputs.shape[0], -1)
        else:
            flattened_inputs = inputs

        # Ensure input size is divisible by num_inputs
        total_elements = flattened_inputs.size
        if total_elements % self.num_inputs != 0:
            raise ValueError(f"Cannot reshape array of size {total_elements} into shape compatible with {self.num_inputs} inputs")
        
        # Reshape the flattened input to match the expected number of inputs for the neuron
        new_shape = (flattened_inputs.shape[0], self.num_inputs)
        reshaped_inputs = flattened_inputs.reshape(new_shape)
        print(f"Reshaped input shape: {reshaped_inputs.shape}")

        # Ensure reshaped input matches the expected weight matrix shape
        if reshaped_inputs.shape[1] != self.num_inputs:
            raise ValueError(f"Input shape {reshaped_inputs.shape[1]} does not match the neuron input size {self.num_inputs}")

        # Compute the weighted sum (z = Wx + b)
        z = np.dot(reshaped_inputs, self.weights) + self.bias
        
        # Apply the quantum activation function
        return self.quantum_activation(z)

class UnifiedQuantumLayer:
    def __init__(self, num_inputs: int, num_neurons: int, entanglement_strength: float = 0.1):
        self.neurons = [UnifiedQuantumNeuron(num_inputs) for _ in range(num_neurons)]
        self.num_neurons = num_neurons
        self.entanglement_strength = entanglement_strength
        self.entanglement_matrix = None

    def generate_entanglement_matrix(self, size: int) -> np.ndarray:
        matrix = np.eye(size) + self.entanglement_strength * np.random.randn(size, size)
        # Ensure normalization respects unitary or orthogonal matrix constraints
        q, _ = np.linalg.qr(matrix)  # Use QR decomposition to generate a unitary matrix
        return q

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        batch_size = inputs.shape[0]
        # Get outputs for each neuron
        print(f"Layer {self}: input shape before forward pass: {inputs.shape}")
        neuron_outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        print(f"Layer {self}: output shape after forward pass: {neuron_outputs.shape}")
        outputs = neuron_outputs.transpose(1, 0, 2)  # Transposed to [batch_size, num_neurons, output_dimension]
        flattened_outputs = outputs.reshape(batch_size, -1)  # Flatten to [batch_size, num_neurons * output_dimension]

        # If entanglement matrix size is incorrect, regenerate it
        if self.entanglement_matrix is None or self.entanglement_matrix.shape[0] != flattened_outputs.shape[1]:
            self.entanglement_matrix = self.generate_entanglement_matrix(flattened_outputs.shape[1])

        entangled_outputs = np.matmul(flattened_outputs, self.entanglement_matrix.T)  # Apply entanglement flattened_output_dimension]
        
        # Reshape entangled outputs back to match expected neuron output shape
        return entangled_outputs.reshape(outputs.shape)

class UnifiedQuantumNeuralNetwork:
    def __init__(self, layer_sizes: List[int], entanglement_strength: float = 0.1):
        if len(layer_sizes) < 2:
            raise ValueError("At least two layer sizes must be provided")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("All layer sizes must be positive integers")
        if not 0 <= entanglement_strength <= 1:
            raise ValueError("Entanglement strength must be between 0 and 1")
        self.logger = setup_logger(self.__class__.__name__, logging.INFO)
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(UnifiedQuantumLayer(layer_sizes[i-1], layer_sizes[i], entanglement_strength))
        self.final_layer_size = layer_sizes[-1]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 2:
            raise ValueError("Input must be a 2D array")
        if inputs.shape[1] != self.layers[0].neurons[0].weights.shape[0]:
            raise ValueError(f"Input shape {inputs.shape} does not match the first layer input size {self.layers[0].neurons[0].weights.shape[0]}")

        self.logger.debug(f"Forward pass, input shape: {inputs.shape}")
        for i, layer in enumerate(self.layers):
            inputs = layer.forward(inputs)
            self.logger.debug(f"After layer {i+1}, shape: {inputs.shape}")
        return inputs.reshape(inputs.shape[0], self.final_layer_size)

    def quantum_attention(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if inputs.ndim != 2:
            raise ValueError("Input must be a 2D array")
        if inputs.shape[1] != self.layers[0].neurons[0].weights.shape[0]:
            raise ValueError(f"Input shape {inputs.shape} does not match the first layer input size {self.layers[0].neurons[0].weights.shape[0]}")

        forward_output = self.forward(inputs)
        attention_weights = np.abs(forward_output)**2
        attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)
        attended_output = np.sum(inputs.reshape(inputs.shape[0], -1)[:, :, np.newaxis] * attention_weights[:, np.newaxis, :], axis=2)
        return attended_output, attention_weights

    def quantum_backpropagation(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epochs: int = 100):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match")
        if X.shape[1] != self.layers[0].neurons[0].weights.shape[0]:
            raise ValueError(f"Input shape {X.shape[1]} does not match the first layer input size {self.layers[0].neurons[0].weights.shape[0]}")
        if y.shape[1] != self.final_layer_size:
            raise ValueError(f"Output shape {y.shape[1]} does not match the final layer size {self.final_layer_size}")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        for epoch in range(epochs):
            total_loss = 0
            for x, target in zip(X, y):
                x = x.reshape(1, -1)
                target = target.reshape(1, -1)
                
                layer_outputs = [x]
                for layer in self.layers:
                    layer_outputs.append(layer.forward(layer_outputs[-1]))

                # Ensure final output matches expected size
                if layer_outputs[-1].size == self.final_layer_size:
                    output = layer_outputs[-1].reshape(1, self.final_layer_size)
                else:
                    raise ValueError(f"Cannot reshape array of size {layer_outputs[-1].size} into shape (1, {self.final_layer_size})")
                loss = np.mean(np.abs(output - target)**2)
                total_loss += loss

                error = output - target
                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    input_data = layer_outputs[i]
                    
                    weight_grad = np.dot(input_data.conj().T, error)
                    bias_grad = np.sum(error, axis=0)
                    
                    for neuron in layer.neurons:
                        anti_hermitian = learning_rate * (weight_grad - weight_grad.conj().T) / 2
                        neuron.weights = np.dot(expm(-1j * anti_hermitian), neuron.weights)
                        neuron.bias -= learning_rate * bias_grad
                    
                    if i > 0:
                        error = np.dot(error, layer.weights.conj().T) * self.quantum_activation_derivative(input_data)

            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {total_loss / len(X)}")


    def quantum_activation_derivative(self, x: np.ndarray) -> np.ndarray:
        return -2 * np.sin(2 * (x + self.layers[0].neurons[0].phase))
