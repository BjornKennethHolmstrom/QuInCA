# main.py

import functools
print = functools.partial(print, flush=True)

from input_system import InputSystem
from cognitive_modules.perception import PerceptionModule
from cognitive_modules.attention import AttentionModule
from cognitive_modules.memory import MemoryModule
from cognitive_modules.reasoning import ReasoningModule
from cognitive_modules.action_selection import ActionSelectionModule
from real_time_framework import RealTimeFramework
from visualization import TerminalVisualizer
import numpy as np

def main():
    print("Starting main function")  # Debug print

    try:
        print("Initializing InputSystem")  # Debug print
        input_system = InputSystem()
        print("InputSystem initialized")  # Debug print

        # Common configuration for all modules
        #input_shape = (64, 36)  # Adjust based on your input
        input_size = 2485  # This is the size of your combined features
        conv_params = {'kernel_size': 3, 'num_filters': 16}
        dim_reduction_size = 128
        attention_size = 64
        layer_sizes = [64, 32, 16, 8]
        memory_size = 100

        print(f"Main configuration:")
        print(f"  input_size: {input_size}")
        print(f"  conv_params: {conv_params}")
        print(f"  dim_reduction_size: {dim_reduction_size}")
        print(f"  attention_size: {attention_size}")
        print(f"  layer_sizes: {layer_sizes}")
        print(f"  memory_size: {memory_size}")

        print("Initializing cognitive modules")  # Debug print
        perception = PerceptionModule(input_size, conv_params, dim_reduction_size, attention_size, layer_sizes)
        print("Perception module initialized")  # Debug print
        attention = AttentionModule(layer_sizes[-1] + 50, conv_params, dim_reduction_size, attention_size, layer_sizes)
        print("Attention module initialized")  # Debug print
        memory = MemoryModule(8 + 50 + memory_size, conv_params, dim_reduction_size, attention_size, layer_sizes, memory_size)
        print("Memory module initialized")  # Debug print
        reasoning = ReasoningModule(16, conv_params, dim_reduction_size, attention_size, layer_sizes)
        print("Reasoning module initialized")  # Debug print
        action_selection = ActionSelectionModule(layer_sizes[-1] + 50, conv_params, dim_reduction_size, attention_size, layer_sizes)
        print("Cognitive modules initialized")  # Debug print

        print("Initializing TerminalVisualizer")  # Debug print
        visualizer = TerminalVisualizer()
        print("TerminalVisualizer initialized")  # Debug print

        # Placeholder for system state
        system_state = np.zeros(50)  # Adjust size as needed

        async def cognitive_cycle():
            print("Starting cognitive cycle")
            try:
                input_data = input_system.get_all_input_data()
                
                # Perception
                perception_output = await perception.process(input_data)
                print(f"Perception output shape: {perception_output.shape}")
                
                # Initialize system state if it doesn't exist
                if 'system_state' not in globals():
                    global system_state
                    system_state = np.zeros((1, 50))  # Adjust size as needed, make it 2D
                
                # Attention
                attended_features, attention_weights = await attention.process(perception_output, system_state)
                print(f"Attended features shape: {attended_features.shape}")
                print(f"Attention weights shape: {attention_weights.shape}")
                
                # Memory
                placeholder_reasoning_output = np.zeros((1, 50))  # Make it 2D
                retrieved_memories = await memory.process(attended_features, placeholder_reasoning_output)
                print(f"Retrieved memories shape: {retrieved_memories.shape}")
                
                # Reasoning
                reasoning_output = await reasoning.process(attended_features, retrieved_memories)
                print(f"Reasoning output shape: {reasoning_output.shape}")
                
                # Action Selection
                selected_action = await action_selection.process(reasoning_output, system_state)
                print(f"Selected action: {selected_action}")
                
                # Update system state (this is a placeholder, replace with actual state update logic)
                system_state = np.random.rand(1, 50)  # Make it 2D
                
                print("Cognitive cycle completed")
            except Exception as e:
                print(f"Error in cognitive_cycle: {e}")
                import traceback
                traceback.print_exc()

        print("Initializing RealTimeFramework")  # Debug print
        framework = RealTimeFramework()
        print("Adding cognitive_cycle task to framework")  # Debug print
        framework.add_task(cognitive_cycle, interval=0.5, name="cognitive_cycle")

        print("Starting framework")  # Debug print
        framework.run()

    except Exception as e:
        print(f"An error occurred in main: {e}")  # Debug print
    finally:
        print("Cleanup complete")  # Debug print

if __name__ == "__main__":
    main()
