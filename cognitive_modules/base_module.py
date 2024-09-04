# cognitive_modules/base_module.py

from quantum_inspired_lib.flexible_quantum_nn import FlexibleQuantumNeuralNetwork
from logger import setup_logger
import logging

class BaseCognitiveModule:
    def __init__(self, initial_input_size, conv_params, dim_reduction_size, attention_size, layer_sizes):
        self.logger = setup_logger(self.__class__.__name__, logging.WARNING)
        self.logger.info("Initializing BaseCognitiveModule")
        try:
            self.logger.debug("Creating FlexibleQuantumNeuralNetwork")
            self.qnn = FlexibleQuantumNeuralNetwork(initial_input_size, conv_params, dim_reduction_size, attention_size, layer_sizes)
            self.logger.debug("FlexibleQuantumNeuralNetwork created")
        except Exception as e:
            self.logger.error(f"Error creating FlexibleQuantumNeuralNetwork: {e}", exc_info=True)
            raise
        self.logger.info("BaseCognitiveModule initialization complete")
    
    async def process(self, input_data):
        # This method should be overridden by subclasses
        raise NotImplementedError
