# tests/test_flexible_quantum_nn.py

import unittest
import numpy as np
import time

from quantum_inspired_lib.flexible_quantum_nn import FlexibleQuantumNeuralNetwork

class TestFlexibleQuantumNN(unittest.TestCase):
    def setUp(self):
        self.initial_input_size = 20
        self.conv_params = {'kernel_size': 3, 'num_filters': 2}
        self.dim_reduction_size = 8
        self.attention_size = 4
        self.layer_sizes = [4, 3, 2]
        self.flexible_nn = FlexibleQuantumNeuralNetwork(
            self.initial_input_size, self.conv_params, self.dim_reduction_size,
            self.attention_size, self.layer_sizes
        )

    def test_initialization(self):
        self.assertIsNotNone(self.flexible_nn.conv_layer)
        self.assertIsNotNone(self.flexible_nn.dim_reduction)
        self.assertIsNotNone(self.flexible_nn.attention)
        self.assertEqual(len(self.flexible_nn.layers), len(self.layer_sizes))

    def test_forward_pass(self):
        input_data = np.random.randn(1, self.initial_input_size)
        output = self.flexible_nn.forward(input_data)
        self.assertEqual(output.shape, (1, 2))

    def test_adjust_input_size(self):
        new_input_size = 30
        self.flexible_nn.adjust_input_size(new_input_size)
        self.assertEqual(self.flexible_nn.conv_layer.input_size, new_input_size)

if __name__ == '__main__':
    unittest.main()
