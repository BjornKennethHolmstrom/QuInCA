# cognitive_modules/reasoning.py

from .base_module import BaseCognitiveModule
import numpy as np

class ReasoningModule(BaseCognitiveModule):
    def __init__(self, initial_input_size, conv_params, dim_reduction_size, attention_size, layer_sizes):
        super().__init__(initial_input_size, conv_params, dim_reduction_size, attention_size, layer_sizes)
        self.logger.info("ReasoningModule initialization complete")

    async def process(self, attended_features, retrieved_memories):
        self.logger.debug("ReasoningModule.process started")
        try:
            # Ensure inputs are 2D arrays
            if len(attended_features.shape) == 1:
                attended_features = attended_features.reshape(1, -1)
            if len(retrieved_memories.shape) == 1:
                retrieved_memories = retrieved_memories.reshape(1, -1)

            self.logger.debug(f"Attended features shape: {attended_features.shape}")
            self.logger.debug(f"Retrieved memories shape: {retrieved_memories.shape}")

            # If retrieved_memories is empty, pad it to match attended_features shape
            if retrieved_memories.shape[1] == 0:
                retrieved_memories = np.zeros_like(attended_features)

            # Concatenate inputs
            combined_input = np.concatenate([attended_features, retrieved_memories], axis=1)
            self.logger.debug(f"Combined input shape: {combined_input.shape}")

            # Adjust the network for the new input size if necessary
            if self.qnn.conv_layer.input_size != combined_input.shape[1]:
                self.qnn.adjust_input_size(combined_input.shape[1])

            # Process through the quantum neural network
            reasoning_output = self.qnn.forward(combined_input)
            self.logger.debug(f"Reasoning output shape: {reasoning_output.shape}")

            self.logger.info("ReasoningModule.process completed")
            return reasoning_output
        except Exception as e:
            self.logger.error(f"Error in ReasoningModule.process: {e}", exc_info=True)
            raise
