# tests/test_unified_quantum_nn.py

import sys
import os
import unittest
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from quantum_inspired_lib.unified_quantum_nn import UnifiedQuantumNeuralNetwork

class TestUnifiedQuantumNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = UnifiedQuantumNeuralNetwork([4, 8, 2])

    def test_initialization(self):
        self.assertEqual(len(self.nn.layers), 2)
        self.assertEqual(self.nn.final_layer_size, 2)

    def test_forward_pass(self):
        input_data = np.random.rand(10, 4)
        output = self.nn.forward(input_data)
        self.assertEqual(output.shape, (10, 2))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            UnifiedQuantumNeuralNetwork([])
        with self.assertRaises(ValueError):
            UnifiedQuantumNeuralNetwork([1, -1, 2])
        with self.assertRaises(ValueError):
            self.nn.forward(np.random.rand(10))
        with self.assertRaises(ValueError):
            self.nn.forward(np.random.rand(10, 5))

    def test_quantum_backpropagation(self):
        X = np.random.rand(20, 4)
        y = np.random.rand(20, 2)
        initial_weights = [np.copy(layer.neurons[0].weights) for layer in self.nn.layers]
        
        self.nn.quantum_backpropagation(X, y, epochs=10)
        
        # Check if weights have been updated
        for i, layer in enumerate(self.nn.layers):
            self.assertFalse(np.allclose(layer.neurons[0].weights, initial_weights[i]))

    def test_unitary_weights(self):
        for layer in self.nn.layers:
            for neuron in layer.neurons:
                w = neuron.weights
                ww_dagger = np.dot(w, w.conj().T)
                self.assertTrue(np.allclose(ww_dagger, np.eye(w.shape[0]), atol=1e-6))

if __name__ == '__main__':
    unittest.main()
