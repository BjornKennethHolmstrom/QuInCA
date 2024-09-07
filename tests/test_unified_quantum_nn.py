# tests/test_unified_quantum_nn.py

import unittest
import numpy as np
from quantum_inspired_lib.unified_quantum_nn import UnifiedQuantumNeuralNetwork

class TestUnifiedQuantumNN(unittest.TestCase):
    def setUp(self):
        self.layer_sizes = [10, 5, 3]
        self.unified_nn = UnifiedQuantumNeuralNetwork(self.layer_sizes)

    def test_initialization(self):
        self.assertEqual(len(self.unified_nn.layers), len(self.layer_sizes) - 1)

    def test_forward_pass(self):
        input_data = np.random.randn(1, 10) + 1j * np.random.randn(1, 10)
        output = self.unified_nn.forward(input_data)
        self.assertEqual(output.shape, (1, 3))
        self.assertTrue(np.iscomplexobj(output))

    def test_quantum_attention(self):
        input_data = np.random.randn(1, 10) + 1j * np.random.randn(1, 10)
        attended_output, attention_weights = self.unified_nn.quantum_attention(input_data)
        self.assertEqual(attended_output.shape, (1, 10))
        self.assertEqual(attention_weights.shape, (1, 3))
        self.assertAlmostEqual(np.sum(attention_weights), 1.0)

    def test_quantum_backpropagation(self):
        X = np.random.randn(100, 10) + 1j * np.random.randn(100, 10)
        y = np.random.randn(100, 3) + 1j * np.random.randn(100, 3)
        initial_loss = np.mean(np.abs(self.unified_nn.forward(X[0:1]) - y[0:1])**2)
        self.unified_nn.quantum_backpropagation(X, y, epochs=10)
        final_loss = np.mean(np.abs(self.unified_nn.forward(X[0:1]) - y[0:1])**2)
        self.assertLess(final_loss, initial_loss)

if __name__ == '__main__':
    unittest.main()
