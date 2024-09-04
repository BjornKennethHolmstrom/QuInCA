# cognitive_modules/perception.py

import logging
from .base_module import BaseCognitiveModule
import numpy as np
from quantum_inspired_lib.efficient_quantum_nn import EfficientQuantumNeuralNetwork
from PIL import Image
from logger import setup_logger

class PerceptionModule(BaseCognitiveModule):
    def __init__(self, input_size, conv_params, dim_reduction_size, attention_size, layer_sizes):
        self.logger = setup_logger(self.__class__.__name__, logging.DEBUG)
        self.logger.info(f"Initializing PerceptionModule")
        self.logger.debug(f"with parameters:")
        self.logger.debug(f"  input_size: {input_size}")
        self.logger.debug(f"  conv_params: {conv_params}")
        self.logger.debug(f"  dim_reduction_size: {dim_reduction_size}")
        self.logger.debug(f"  attention_size: {attention_size}")
        self.logger.debug(f"  layer_sizes: {layer_sizes}")
        
        self.input_size = input_size
        self.conv_params = conv_params
        self.dim_reduction_size = dim_reduction_size
        self.attention_size = attention_size
        self.layer_sizes = layer_sizes
        self.qnn = EfficientQuantumNeuralNetwork(
            input_size=input_size,
            conv_params=conv_params,
            dim_reduction_size=dim_reduction_size,
            attention_size=attention_size,
            layer_sizes=layer_sizes
        )
        self.logger.info("PerceptionModule initialization complete")
    
    async def process(self, input_data):
        self.logger.debug("PerceptionModule.process started")
        try:
            # Preprocess inputs
            audio_features = self.preprocess_audio(input_data.get('audio', np.array([])))
            keyboard_features = self.preprocess_keyboard(input_data.get('keyboard', []))
            mouse_features = self.preprocess_mouse(input_data.get('mouse', []))
            screenshot = self.preprocess_screenshot(input_data.get('screenshot', np.array([])))
            
            self.logger.debug(f"Preprocessed features shapes:")
            self.logger.debug(f"  Audio: {audio_features.shape}")
            self.logger.debug(f"  Keyboard: {keyboard_features.shape}")
            self.logger.debug(f"  Mouse: {mouse_features.shape}")
            self.logger.debug(f"  Screenshot: {screenshot.shape}")
            
            # Flatten the screenshot
            screenshot_flat = screenshot.reshape(screenshot.shape[0], -1)
            
            # Combine all features
            combined_features = np.concatenate([audio_features, keyboard_features, mouse_features, screenshot_flat], axis=1)
            
            self.logger.debug(f"Combined features shape: {combined_features.shape}")
            
            # Process through the quantum neural network
            visual_features = self.qnn.forward(combined_features)
            
            self.logger.debug("PerceptionModule.process completed")
            return visual_features
        except Exception as e:
            self.logger.error(f"Error in PerceptionModule.process: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def preprocess_audio(self, audio_data):
        self.logger.debug("Preprocessing audio")
        if len(audio_data) > 0:
            features = np.abs(np.fft.fft(audio_data))[:50]
            return features.reshape(1, -1)
        return np.zeros((1, 50))
    
    def preprocess_keyboard(self, keyboard_events):
        self.logger.debug("Preprocessing keyboard")
        key_vector = np.zeros(128)  # Assuming ASCII
        for event, key in keyboard_events:
            if hasattr(key, 'vk'):
                key_vector[key.vk] = 1 if event == 'press' else 0
        return key_vector.reshape(1, -1)
    
    def preprocess_mouse(self, mouse_events):
        self.logger.debug("Preprocessing mouse")
        if mouse_events:
            last_event = mouse_events[-1]
            if last_event[0] == 'move':
                return np.array([[last_event[1][0], last_event[1][1], 0]])
            elif last_event[0] == 'click':
                return np.array([[last_event[1][0], last_event[1][1], 1 if last_event[1][3] else 0]])
        return np.zeros((1, 3))

    def preprocess_screenshot(self, screenshot_data):
        self.logger.debug("Preprocessing screenshot")
        if screenshot_data.size > 0:
            img = Image.fromarray(screenshot_data)
            img_resized = img.resize((64, 36))  # Match the input shape expected by the network
            img_gray = img_resized.convert('L')
            return np.array(img_gray).reshape(1, 64, 36) / 255.0  # Normalize to [0, 1]
        else:
            return np.zeros((1, 64, 36))
