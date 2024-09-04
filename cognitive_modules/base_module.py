# cognitive_modules/base_module.py

import logging
from quantum_inspired_lib.efficient_quantum_nn import EfficientQuantumNeuralNetwork
from logger import setup_logger

class BaseCognitiveModule:
    def __init__(self, input_shape, conv_params, dim_reduction_size, attention_size, layer_sizes):
        self.logger = setup_logger(self.__class__.__name__, logging.DEBUG)
        self.logger.info("Initializing BaseCognitiveModule")
        try:
            self.logger.debug("Creating EfficientQuantumNeuralNetwork")
            self.qnn = EfficientQuantumNeuralNetwork(input_shape, conv_params, dim_reduction_size, attention_size, layer_sizes)
            self.logger.debug("EfficientQuantumNeuralNetwork created")
        except Exception as e:
            self.logger.error(f"Error creating EfficientQuantumNeuralNetwork: {e}", exc_info=True)
            raise
        self.logger.info("BaseCognitiveModule initialization complete")
    
    async def process(self, input_data):
        # This method should be overridden by subclasses
        raise NotImplementedError
