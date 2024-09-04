# cognitive_modules/memory.py

import logging
from .base_module import BaseCognitiveModule
from quantum_inspired_lib.efficient_quantum_nn import DimensionalityReduction, ConvolutionalLayer
import numpy as np
from logger import setup_logger

class MemoryModule(BaseCognitiveModule):
    def __init__(self, input_size, conv_params, dim_reduction_size, attention_size, layer_sizes, memory_size):
        self.logger = setup_logger(self.__class__.__name__, logging.DEBUG)
        self.logger.info("Initializing MemoryModule")
        # Calculate the correct input size for the memory module
        actual_input_size = 8 + 50 + memory_size  # attended_features + reasoning_output + memory
        
        # Adjust conv_params for the new input size
        adjusted_conv_params = conv_params.copy()
        adjusted_conv_params['kernel_size'] = min(conv_params['kernel_size'], actual_input_size - 1)
        
        super().__init__(actual_input_size, adjusted_conv_params, dim_reduction_size, attention_size, layer_sizes)
        self.memory = np.zeros((1, memory_size))  # Initialize memory as a 2D array
        self.logger.info("MemoryModule initialization complete")

    async def process(self, attended_features, reasoning_output):
        self.logger.debug("MemoryModule.process started")
        try:
            # Ensure all inputs are 2D arrays
            if len(attended_features.shape) == 1:
                attended_features = attended_features.reshape(1, -1)
            if len(reasoning_output.shape) == 1:
                reasoning_output = reasoning_output.reshape(1, -1)

            self.logger.debug(f"Attended features shape: {attended_features.shape}")
            self.logger.debug(f"Reasoning output shape: {reasoning_output.shape}")
            self.logger.debug(f"Memory shape: {self.memory.shape}")

            # Concatenate inputs
            combined_input = np.concatenate([attended_features, reasoning_output, self.memory], axis=1)
            self.logger.debug(f"Combined input shape: {combined_input.shape}")

            # Adjust the layers if necessary
            if self.qnn.conv_layer.input_size != combined_input.shape[1]:
                self.logger.debug("Adjusting ConvolutionalLayer and DimensionalityReduction layer")
                new_conv_params = self.qnn.conv_layer.__dict__.copy()
                new_conv_params['input_size'] = combined_input.shape[1]
                # Remove unexpected arguments
                new_conv_params.pop('logger', None)
                new_conv_params.pop('kernels', None)
                self.qnn.conv_layer = ConvolutionalLayer(**new_conv_params)
                conv_output_size = self.qnn.conv_layer.num_filters * (combined_input.shape[1] - new_conv_params['kernel_size'] + 1)
                self.qnn.dim_reduction = DimensionalityReduction(conv_output_size, self.qnn.dim_reduction.output_size)

            # Process through the quantum neural network
            memory_output = self.qnn.forward(combined_input)
            self.logger.debug(f"Memory output shape: {memory_output.shape}")

            # Update memory and return retrieved memories
            self.memory = memory_output[:, :self.memory.shape[1]]
            retrieved_memories = memory_output[:, self.memory.shape[1]:]

            self.logger.debug("MemoryModule.process completed")
            return retrieved_memories
        except Exception as e:
            self.logger.error(f"Error in MemoryModule.process: {e}", exc_info=True)
            raise
