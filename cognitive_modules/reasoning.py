# cognitive_modules/reasoning.py

from .base_module import BaseCognitiveModule
import numpy as np

class ReasoningModule(BaseCognitiveModule):
    def __init__(self, input_size, conv_params, dim_reduction_size, attention_size, layer_sizes):
        print("Initializing ReasoningModule")
        super().__init__(input_size, conv_params, dim_reduction_size, attention_size, layer_sizes)
        print("ReasoningModule initialization complete")

    async def process(self, attended_features, retrieved_memories):
        print("ReasoningModule.process started")
        try:
            # Ensure inputs are 2D arrays
            if len(attended_features.shape) == 1:
                attended_features = attended_features.reshape(1, -1)
            if len(retrieved_memories.shape) == 1:
                retrieved_memories = retrieved_memories.reshape(1, -1)

            print(f"Attended features shape: {attended_features.shape}")
            print(f"Retrieved memories shape: {retrieved_memories.shape}")

            # If retrieved_memories is empty, pad it to match attended_features shape
            if retrieved_memories.shape[1] == 0:
                retrieved_memories = np.zeros_like(attended_features)

            # Concatenate inputs
            combined_input = np.concatenate([attended_features, retrieved_memories], axis=1)
            print(f"Combined input shape: {combined_input.shape}")

            # Process through the quantum neural network
            reasoning_output = self.qnn.forward(combined_input)
            print(f"Reasoning output shape: {reasoning_output.shape}")

            print("ReasoningModule.process completed")
            return reasoning_output
        except Exception as e:
            print(f"Error in ReasoningModule.process: {e}")
            import traceback
            traceback.print_exc()
            raise
