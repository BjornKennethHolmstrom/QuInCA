# cognitive_modules/attention.py

from .base_module import BaseCognitiveModule
import numpy as np

class AttentionModule(BaseCognitiveModule):
    def __init__(self, initial_input_size, conv_params, dim_reduction_size, attention_size, layer_sizes):
        super().__init__(initial_input_size, conv_params, dim_reduction_size, attention_size, layer_sizes)
        self.logger.info("AttentionModule initialization complete")

    async def process(self, features, current_state):
        self.logger.debug("AttentionModule.process started")
        try:
            # Ensure features and current_state are 2D
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            if len(current_state.shape) == 1:
                current_state = current_state.reshape(1, -1)

            self.logger.debug(f"Features shape: {features.shape}")
            self.logger.debug(f"Current state shape: {current_state.shape}")

            # Concatenate features and current_state
            combined_input = np.concatenate([features, current_state], axis=1)
            self.logger.debug(f"Combined input shape: {combined_input.shape}")

            # Adjust the network for the new input size if necessary
            if self.qnn.conv_layer.input_size != combined_input.shape[1]:
                self.qnn.adjust_input_size(combined_input.shape[1])

            # Process through the quantum neural network
            attention_output = self.qnn.forward(combined_input)
            self.logger.debug(f"Attention output shape: {attention_output.shape}")

            # Split output into attended features and attention weights
            split_index = features.shape[1]
            attended_features = attention_output[:, :split_index]
            attention_weights = attention_output[:, split_index:]

            self.logger.info("AttentionModule.process completed")
            return attended_features, attention_weights
        except Exception as e:
            self.logger.error(f"Error in AttentionModule.process: {e}", exc_info=True)
            raise
