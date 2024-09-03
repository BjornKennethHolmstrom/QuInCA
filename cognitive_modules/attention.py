# cognitive_modules/attention.py

from .base_module import BaseCognitiveModule
import numpy as np

class AttentionModule(BaseCognitiveModule):
    def __init__(self, input_size, conv_params, dim_reduction_size, attention_size, layer_sizes):
        print("Initializing AttentionModule")
        # Calculate the correct input size for the attention module
        perception_output_size = layer_sizes[-1]  # Size of the last layer in perception module
        state_size = 50  # Size of the system state
        actual_input_size = perception_output_size + state_size
        
        # Adjust conv_params for the new input size
        adjusted_conv_params = conv_params.copy()
        adjusted_conv_params['kernel_size'] = min(conv_params['kernel_size'], actual_input_size - 1)
        
        super().__init__(actual_input_size, adjusted_conv_params, dim_reduction_size, attention_size, layer_sizes)
        print("AttentionModule initialization complete")

    async def process(self, features, current_state):
        print("AttentionModule.process started")
        try:
            # Ensure features and current_state are 2D
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            if len(current_state.shape) == 1:
                current_state = current_state.reshape(1, -1)

            print(f"Features shape: {features.shape}")
            print(f"Current state shape: {current_state.shape}")

            # Concatenate features and current_state
            combined_input = np.concatenate([features, current_state], axis=1)
            print(f"Combined input shape: {combined_input.shape}")

            # Process through the quantum neural network
            attention_output = self.qnn.forward(combined_input)
            print(f"Attention output shape: {attention_output.shape}")

            # Split output into attended features and attention weights
            split_index = features.shape[1]
            attended_features = attention_output[:, :split_index]
            attention_weights = attention_output[:, split_index:]

            print("AttentionModule.process completed")
            return attended_features, attention_weights
        except Exception as e:
            print(f"Error in AttentionModule.process: {e}")
            import traceback
            traceback.print_exc()
            raise
