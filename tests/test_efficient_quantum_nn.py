# tests/test_efficient_quantum_nn.py

import unittest
import numpy as np
import time

from quantum_inspired_lib.efficient_quantum_nn import EfficientQuantumNeuralNetwork

class TestEfficientQuantumNN(unittest.TestCase):
    def setUp(self):
        self.input_size = 20
        self.conv_params = {'kernel_size': 3, 'num_filters': 2}
        self.dim_reduction_size = 8
        self.attention_size = 4
        self.layer_sizes = [4, 3, 2]
        self.efficient_nn = EfficientQuantumNeuralNetwork(
            self.input_size, self.conv_params, self.dim_reduction_size,
            self.attention_size, self.layer_sizes
        )

    def test_initialization(self):
        self.assertIsNotNone(self.efficient_nn.conv_layer)
        self.assertIsNotNone(self.efficient_nn.dim_reduction)
        self.assertIsNotNone(self.efficient_nn.attention)
        self.assertEqual(len(self.efficient_nn.layers), len(self.layer_sizes))

    def test_forward_pass(self):
        input_data = np.random.randn(1, self.input_size)
        output = self.efficient_nn.forward(input_data)
        self.assertEqual(output.shape, (1, 2))

    def test_efficiency(self):
        input_data = np.random.randn(100, self.input_size)
        start_time = time.time()
        _ = self.efficient_nn.forward(input_data)
        end_time = time.time()
        self.assertLess(end_time - start_time, 1.0)  # Assuming it should take less than 1 second

if __name__ == '__main__':
    unittest.main()
