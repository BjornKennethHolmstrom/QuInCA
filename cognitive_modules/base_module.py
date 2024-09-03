# cognitive_modules/base_module.py

from quantum_inspired_lib.efficient_quantum_nn import EfficientQuantumNeuralNetwork
print = __import__('functools').partial(print, flush=True)

class BaseCognitiveModule:
    def __init__(self, input_shape, conv_params, dim_reduction_size, attention_size, layer_sizes):
        print("Initializing BaseCognitiveModule")  # Debug print
        try:
            print("Creating EfficientQuantumNeuralNetwork")  # Debug print
            self.qnn = EfficientQuantumNeuralNetwork(input_shape, conv_params, dim_reduction_size, attention_size, layer_sizes)
            print("EfficientQuantumNeuralNetwork created")  # Debug print
        except Exception as e:
            print(f"Error creating EfficientQuantumNeuralNetwork: {e}")  # Debug print
            raise
        print("BaseCognitiveModule initialization complete")  # Debug print
    
    async def process(self, input_data):
        # This method should be overridden by subclasses
        raise NotImplementedError
