# visualization.py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import time
import os

class SystemVisualizer:
    def __init__(self, num_modules=5):
        self.num_modules = num_modules
        self.fig, self.axs = plt.subplots(num_modules, 1, figsize=(10, 3*num_modules))
        self.lines = [ax.plot([], [])[0] for ax in self.axs]
        self.data = [[] for _ in range(num_modules)]
        
        for ax in self.axs:
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)
        
        self.axs[0].set_title("Perception")
        self.axs[1].set_title("Attention")
        self.axs[2].set_title("Memory")
        self.axs[3].set_title("Reasoning")
        self.axs[4].set_title("Action Selection")

    def update_data(self, module_index, new_data):
        self.data[module_index].append(np.mean(new_data))
        if len(self.data[module_index]) > 100:
            self.data[module_index] = self.data[module_index][-100:]

    def animate(self, frame):
        for i, line in enumerate(self.lines):
            line.set_data(range(len(self.data[i])), self.data[i])
        return self.lines

    def start_visualization(self):
        self.anim = FuncAnimation(self.fig, self.animate, frames=200, interval=50, blit=True)
        plt.show(block=False)
        plt.pause(0.1)

    def update_plot(self):
        for i, line in enumerate(self.lines):
            line.set_data(range(len(self.data[i])), self.data[i])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class TerminalVisualizer:
    def __init__(self):
        self.module_names = ["Perception", "Attention", "Memory", "Reasoning", "Action", "System State"]
        self.data = {name: (0, 0) for name in self.module_names}  # (value, shape)
        self.max_width = 50  # Maximum width of the bar

    def update(self, new_data):
        for key, value in new_data.items():
            if key in self.data:
                if isinstance(value, (int, float)):
                    self.data[key] = (value, None)
                elif hasattr(value, 'shape'):
                    self.data[key] = (self.calculate_complexity(value.shape), value.shape)
                else:
                    self.data[key] = (0, str(value))

    def calculate_complexity(self, shape):
        # Simple complexity measure: sum of dimensions
        return sum(shape) / 100  # Normalize to 0-1 range

    def generate_bar(self, value):
        filled_width = int(value * self.max_width)
        return f"[{'#' * filled_width}{'-' * (self.max_width - filled_width)}]"

    def clear_terminal(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def render(self):
        self.clear_terminal()
        print("Quantum-Inspired Cognitive Architecture Visualization")
        print("=" * 70)
        for name in self.module_names:
            value, shape = self.data[name]
            bar = self.generate_bar(value)
            shape_str = f"Shape: {shape}" if shape else ""
            print(f"{name:12} {bar} {value:.2f} {shape_str}")
        print("=" * 70)
        print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
