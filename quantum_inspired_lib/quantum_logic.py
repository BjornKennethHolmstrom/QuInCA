import numpy as np

class QuantumRegister:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1  # Initialize to |00...0>

    def measure(self):
        probabilities = np.abs(self.state)**2
        result = np.random.choice(2**self.num_qubits, p=probabilities)
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[result] = 1
        return result

class QuantumGates:
    @staticmethod
    def hadamard(register, qubit):
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        QuantumGates._apply_single_qubit_gate(register, H, qubit)

    @staticmethod
    def pauli_x(register, qubit):
        X = np.array([[0, 1], [1, 0]])
        QuantumGates._apply_single_qubit_gate(register, X, qubit)

    @staticmethod
    def pauli_y(register, qubit):
        Y = np.array([[0, -1j], [1j, 0]])
        QuantumGates._apply_single_qubit_gate(register, Y, qubit)

    @staticmethod
    def pauli_z(register, qubit):
        Z = np.array([[1, 0], [0, -1]])
        QuantumGates._apply_single_qubit_gate(register, Z, qubit)

    @staticmethod
    def cnot(register, control, target):
        CNOT = np.eye(4)
        CNOT[2:, 2:] = np.array([[0, 1], [1, 0]])
        QuantumGates._apply_two_qubit_gate(register, CNOT, control, target)

    @staticmethod
    def _apply_single_qubit_gate(register, gate, qubit):
        n = register.num_qubits
        full_gate = np.eye(2**n, dtype=complex)
        for i in range(2**n):
            if i & (1 << (n - qubit - 1)):
                i2 = i ^ (1 << (n - qubit - 1))
                full_gate[i, i] = gate[1, 1]
                full_gate[i, i2] = gate[1, 0]
                full_gate[i2, i] = gate[0, 1]
                full_gate[i2, i2] = gate[0, 0]
        register.state = np.dot(full_gate, register.state)

    @staticmethod
    def _apply_two_qubit_gate(register, gate, qubit1, qubit2):
        n = register.num_qubits
        full_gate = np.eye(2**n, dtype=complex)
        for i in range(2**n):
            i1 = i & (1 << (n - qubit1 - 1))
            i2 = i & (1 << (n - qubit2 - 1))
            index = (i1 >> (n - qubit1 - 1)) | ((i2 >> (n - qubit2 - 1)) << 1)
            for j in range(4):
                if gate[index, j] != 0:
                    i_new = i ^ ((j & 1) << (n - qubit1 - 1)) ^ ((j >> 1) << (n - qubit2 - 1))
                    full_gate[i_new, i] = gate[index, j]
        register.state = np.dot(full_gate, register.state)

# Example usage
if __name__ == "__main__":
    # Create a 2-qubit register
    qr = QuantumRegister(2)

    # Apply Hadamard gate to the first qubit
    QuantumGates.hadamard(qr, 0)

    # Apply CNOT gate with first qubit as control and second as target
    QuantumGates.cnot(qr, 0, 1)

    # Measure the state
    result = qr.measure()
    print(f"Measured state: {result:02b}")
