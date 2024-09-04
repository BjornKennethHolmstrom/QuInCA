from .base_module import BaseCognitiveModule
from quantum_inspired_lib.flexible_quantum_nn import FlexibleQuantumNeuralNetwork
import numpy as np

class ActionSelectionModule(BaseCognitiveModule):
    def __init__(self, input_size, conv_params, dim_reduction_size, attention_size, layer_sizes):
        super().__init__(input_size, conv_params, dim_reduction_size, attention_size, layer_sizes)
        self.qnn = FlexibleQuantumNeuralNetwork(input_size, conv_params, dim_reduction_size, attention_size, layer_sizes)

    async def process(self, reasoning_output, current_state):
        self.logger.debug("ActionSelectionModule.process started")
        try:
            # Ensure inputs are 2D arrays
            if len(reasoning_output.shape) == 1:
                reasoning_output = reasoning_output.reshape(1, -1)
            if len(current_state.shape) == 1:
                current_state = current_state.reshape(1, -1)

            self.logger.debug(f"Reasoning output shape: {reasoning_output.shape}")
            self.logger.debug(f"Current state shape: {current_state.shape}")

            # Concatenate inputs
            combined_input = np.concatenate([reasoning_output, current_state], axis=1)
            self.logger.debug(f"Combined input shape: {combined_input.shape}")

            # Adjust the network for the new input size if necessary
            if self.qnn.conv_layer.input_size != combined_input.shape[1]:
                self.qnn.adjust_input_size(combined_input.shape[1])

            # Process through the quantum neural network
            action_probabilities = self.qnn.forward(combined_input)
            self.logger.debug(f"Action probabilities shape: {action_probabilities.shape}")

            # Select action with highest probability
            selected_action = np.argmax(action_probabilities)

            self.logger.info("ActionSelectionModule.process completed")
            return selected_action
        except Exception as e:
            self.logger.error(f"Error in ActionSelectionModule.process: {e}", exc_info=True)
            raise
