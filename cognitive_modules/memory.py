# cognitive_modules/memory.py

from .base_module import BaseCognitiveModule
import numpy as np

class MemoryModule(BaseCognitiveModule):
    def __init__(self, initial_input_size, conv_params, dim_reduction_size, attention_size, layer_sizes, memory_size):
        super().__init__(initial_input_size, conv_params, dim_reduction_size, attention_size, layer_sizes)
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

            # Adjust the network for the new input size if necessary
            if self.qnn.conv_layer.input_size != combined_input.shape[1]:
                self.qnn.adjust_input_size(combined_input.shape[1])

            # Process through the quantum neural network
            memory_output = self.qnn.forward(combined_input)
            self.logger.debug(f"Memory output shape: {memory_output.shape}")

            # Update memory and return retrieved memories
            self.memory = memory_output[:, :self.memory.shape[1]]
            retrieved_memories = memory_output[:, self.memory.shape[1]:]

            self.logger.info("MemoryModule.process completed")
            return retrieved_memories
        except Exception as e:
            self.logger.error(f"Error in MemoryModule.process: {e}", exc_info=True)
            raise
