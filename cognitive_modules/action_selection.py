# cognitive_modules/action_selection.py

import logging
from .base_module import BaseCognitiveModule
from quantum_inspired_lib.efficient_quantum_nn import DimensionalityReduction, ConvolutionalLayer
import numpy as np
from logger import setup_logger

class ActionSelectionModule(BaseCognitiveModule):
    def __init__(self, input_size, conv_params, dim_reduction_size, attention_size, layer_sizes):
        self.logger = setup_logger(self.__class__.__name__, logging.DEBUG)
        self.logger.info("Initializing ActionSelectionModule")
        super().__init__(input_size, conv_params, dim_reduction_size, attention_size, layer_sizes)
        self.logger.debug("ActionSelectionModule initialization complete")

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

            # Pad reasoning_output to match current_state size
            padded_reasoning = np.pad(reasoning_output, ((0, 0), (0, current_state.shape[1] - reasoning_output.shape[1])))

            # Concatenate inputs
            combined_input = np.concatenate([padded_reasoning, current_state], axis=1)
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
            action_probabilities = self.qnn.forward(combined_input)
            self.logger.debug(f"Action probabilities shape: {action_probabilities.shape}")

            # Select action with highest probability
            selected_action = np.argmax(action_probabilities)

            self.logger.info("ActionSelectionModule.process completed")
            return selected_action
        except Exception as e:
            self.logger.error(f"Error in ActionSelectionModule.process: {e}")
            import traceback
            traceback.print_exc()
            raise
