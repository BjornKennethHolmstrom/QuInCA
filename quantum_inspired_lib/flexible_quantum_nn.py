import numpy as np
from logger import setup_logger
import logging

class FlexibleQuantumNeuralNetwork:
    def __init__(self, initial_input_size, conv_params, dim_reduction_size, attention_size, layer_sizes):
        self.logger = setup_logger(self.__class__.__name__, logging.WARNING)
        self.conv_params = conv_params
        self.dim_reduction_size = dim_reduction_size
        self.attention_size = attention_size
        self.layer_sizes = layer_sizes
        
        self.conv_layer = FlexibleConvolutionalLayer(initial_input_size, **conv_params)
        self.dim_reduction = FlexibleDimensionalityReduction(self.conv_layer.output_size, dim_reduction_size)
        self.attention = FlexibleAttentionMechanism(dim_reduction_size, attention_size)
        
        self.quantum_layers = []
        current_size = attention_size
        for size in layer_sizes:
            self.quantum_layers.append(FlexibleQuantumLayer(current_size, size))
            current_size = size
    
    def forward(self, inputs):
        self.logger.debug(f"Forward pass, input shape: {inputs.shape}")
        x = self.conv_layer.forward(inputs)
        self.logger.debug(f"After conv layer, shape: {x.shape}")
        x = x.reshape(x.shape[0], -1)  # Flatten the output
        self.logger.debug(f"After reshaping, shape: {x.shape}")
        
        # Adjust dimensionality reduction if needed
        if x.shape[1] != self.dim_reduction.input_size:
            self.dim_reduction.adjust_input_size(x.shape[1])
        
        x = self.dim_reduction.forward(x)
        self.logger.debug(f"After dim reduction, shape: {x.shape}")
        x = self.attention.forward(x)
        self.logger.debug(f"After attention, shape: {x.shape}")
        for i, layer in enumerate(self.quantum_layers):
            x = layer.forward(x)
            self.logger.debug(f"After quantum layer {i+1}, shape: {x.shape}")
        return x
    
    def adjust_input_size(self, new_input_size):
        self.logger.info(f"Adjusting network for new input size: {new_input_size}")
        self.conv_layer.adjust_input_size(new_input_size)
        new_dim_reduction_input_size = self.conv_layer.output_size * self.conv_params['num_filters']
        self.dim_reduction.adjust_input_size(new_dim_reduction_input_size)

class FlexibleConvolutionalLayer:
    def __init__(self, input_size, kernel_size, num_filters):
        self.logger = setup_logger(self.__class__.__name__, logging.WARNING)
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.kernels = np.random.randn(num_filters, kernel_size) / np.sqrt(kernel_size)
        self.output_size = input_size - kernel_size + 1
    
    def forward(self, inputs):
        self.logger.debug(f"ConvolutionalLayer input shape: {inputs.shape}")
        batch_size, input_size = inputs.shape
        output_size = input_size - self.kernel_size + 1
        output = np.zeros((batch_size, self.num_filters, output_size))

        for b in range(batch_size):
            for i in range(output_size):
                input_slice = inputs[b, i:i+self.kernel_size]
                for f in range(self.num_filters):
                    output[b, f, i] = np.sum(input_slice * self.kernels[f])
        
        self.logger.debug(f"ConvolutionalLayer output shape: {output.shape}")
        return output.reshape(batch_size, -1)  # Flatten the output
    
    def adjust_input_size(self, new_input_size):
        self.input_size = new_input_size
        self.output_size = new_input_size - self.kernel_size + 1

class FlexibleDimensionalityReduction:
    def __init__(self, input_size, output_size):
        self.logger = setup_logger(self.__class__.__name__, logging.WARNING)
        self.input_size = input_size
        self.output_size = output_size
        self.projection_matrix = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.logger.debug(f"Initialized with input_size={input_size}, output_size={output_size}")
    
    def forward(self, inputs):
        if inputs.shape[1] != self.input_size:
            self.logger.warning(f"Input size mismatch. Expected {self.input_size}, got {inputs.shape[1]}. Adjusting.")
            self.adjust_input_size(inputs.shape[1])
        return np.dot(inputs, self.projection_matrix)
    
    def adjust_input_size(self, new_input_size):
        self.logger.info(f"Adjusting input size from {self.input_size} to {new_input_size}")
        self.input_size = new_input_size
        self.projection_matrix = np.random.randn(new_input_size, self.output_size) / np.sqrt(new_input_size)

class FlexibleAttentionMechanism:
    def __init__(self, input_size, output_size):
        self.logger = setup_logger(self.__class__.__name__, logging.WARNING)
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.v = np.random.randn(output_size) / np.sqrt(output_size)
    
    def forward(self, inputs):
        self.logger.debug(f"AttentionMechanism input shape: {inputs.shape}")
        
        # Ensure inputs is 2D
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        
        # Compute attention scores
        u = np.tanh(np.dot(inputs, self.W))
        attention_scores = np.exp(np.dot(u, self.v))
        
        # Handle the case where attention_scores is 1D
        if len(attention_scores.shape) == 1:
            attention_scores = attention_scores.reshape(1, -1)
        
        attention_weights = attention_scores / np.sum(attention_scores, axis=1, keepdims=True)
        
        # Apply attention weights
        weighted_inputs = inputs * attention_weights
        
        # Sum along the feature dimension
        output = np.sum(weighted_inputs, axis=1)
        
        # Ensure output matches the expected output size
        if output.shape[0] != self.output_size:
            if output.shape[0] > self.output_size:
                output = output[:self.output_size]
            else:
                output = np.pad(output, (0, self.output_size - output.shape[0]))
        
        output = output.reshape(1, -1)  # Ensure output is 2D
        
        self.logger.debug(f"AttentionMechanism output shape: {output.shape}")
        return output

class FlexibleQuantumLayer:
    def __init__(self, input_size, output_size, sparsity=0.1):
        self.logger = setup_logger(self.__class__.__name__, logging.WARNING)
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [QuantumNeuron(input_size) for _ in range(output_size)]
        self.connectivity_mask = np.random.choice([0, 1], size=(output_size, input_size), p=[1-sparsity, sparsity])
    
    def forward(self, inputs):
        self.logger.debug(f"QuantumLayer input shape: {inputs.shape}")
        
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        
        if inputs.shape[1] != self.input_size:
            self.logger.warning(f"Input size mismatch. Expected {self.input_size}, got {inputs.shape[1]}")
            # Adjust input size if necessary
            if inputs.shape[1] > self.input_size:
                inputs = inputs[:, :self.input_size]
            else:
                inputs = np.pad(inputs, ((0, 0), (0, self.input_size - inputs.shape[1])))
        
        outputs = np.zeros((inputs.shape[0], self.output_size))
        for i, (neuron, mask) in enumerate(zip(self.neurons, self.connectivity_mask)):
            masked_inputs = inputs * mask
            outputs[:, i] = neuron.forward(masked_inputs).flatten()
        
        self.logger.debug(f"QuantumLayer output shape: {outputs.shape}")
        return outputs

class QuantumNeuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs) / np.sqrt(num_inputs)
        self.bias = np.random.randn()
        self.phase = np.random.uniform(0, 2*np.pi)

    def quantum_activation(self, x):
        return np.cos(x + self.phase) ** 2

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.quantum_activation(z)
