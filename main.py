import asyncio
import logging
from logger import setup_logger
from input_system import InputSystem
from cognitive_modules.perception import PerceptionModule
from cognitive_modules.attention import AttentionModule
from cognitive_modules.memory import MemoryModule
from cognitive_modules.reasoning import ReasoningModule
from cognitive_modules.action_selection import ActionSelectionModule
from real_time_framework import RealTimeFramework
from visualization import TerminalVisualizer
import numpy as np

# Set up logger for main
logger = setup_logger('main', logging.INFO)

# Define actions
ACTIONS = ["move_forward", "turn_left", "turn_right", "wait", "interact"]

# Global variables
system_state = None
previous_reasoning_output = None
input_system = None
perception = None
attention = None
memory = None
reasoning = None
action_selection = None
visualizer = None

async def cognitive_cycle():
    global system_state, previous_reasoning_output, input_system
    logger.debug("Starting cognitive cycle")
    try:
        # Initialize system_state if it doesn't exist
        if system_state is None:
            system_state = np.zeros((1, 50))  # Adjust size as needed, make it 2D
        
        # Initialize previous_reasoning_output if it doesn't exist
        if previous_reasoning_output is None:
            previous_reasoning_output = np.zeros((1, 50))  # Adjust size as needed

        input_data = input_system.get_all_input_data()
        
        # Perception
        perception_output = await perception.process(input_data)
        logger.debug(f"Perception output shape: {perception_output.shape}")
        
        # Attention
        attended_features, attention_weights = await attention.process(perception_output, system_state)
        logger.debug(f"Attended features shape: {attended_features.shape}")
        logger.debug(f"Attention weights shape: {attention_weights.shape}")
        
        # Memory
        retrieved_memories = await memory.process(attended_features, previous_reasoning_output)
        logger.debug(f"Retrieved memories shape: {retrieved_memories.shape}")
        
        # Reasoning
        reasoning_output = await reasoning.process(attended_features, retrieved_memories)
        logger.debug(f"Reasoning output shape: {reasoning_output.shape}")
        
        # Action Selection
        action_probabilities = await action_selection.process(reasoning_output, system_state)
        selected_action = np.argmax(action_probabilities)
        logger.info(f"Selected action: {ACTIONS[selected_action]}")
        
        # Update system state
        new_state = np.concatenate([
            np.mean(attended_features, axis=0, keepdims=True),  # Average attended features
            np.mean(retrieved_memories, axis=0, keepdims=True),  # Average retrieved memories
            reasoning_output,  # Current reasoning output
            np.array([[selected_action]])  # Selected action
        ], axis=1)

        # Ensure the state has a consistent size
        target_size = 50
        if new_state.shape[1] < target_size:
            new_state = np.pad(new_state, ((0, 0), (0, target_size - new_state.shape[1])))
        elif new_state.shape[1] > target_size:
            new_state = new_state[:, :target_size]

        system_state = new_state
        logger.debug(f"Updated system state shape: {system_state.shape}")

        # Update previous_reasoning_output for the next cycle
        previous_reasoning_output = reasoning_output

        # Visualization
        visualizer.update({
            "Perception": perception_output,
            "Attention": attended_features,
            "Memory": retrieved_memories,
            "Reasoning": reasoning_output,
            "Action": ACTIONS[selected_action],
            "System State": system_state
        })
        visualizer.render()
        
        logger.debug("Cognitive cycle completed")
    except Exception as e:
        logger.error(f"Error in cognitive_cycle: {e}", exc_info=True)

async def main():
    logger.info("Starting main function")

    try:
        global input_system, perception, attention, memory, reasoning, action_selection, visualizer
        
        logger.info("Initializing InputSystem")
        input_system = InputSystem()
        logger.info("InputSystem initialized")

        # Common configuration for all modules
        initial_input_size = 2485  # This is the initial size of your combined features
        conv_params = {'kernel_size': 3, 'num_filters': 16}
        dim_reduction_size = 128
        attention_size = 64
        layer_sizes = [64, 32, 16, 8]
        memory_size = 100

        logger.info(f"Main configuration: initial_input_size={initial_input_size}, conv_params={conv_params}, "
                    f"dim_reduction_size={dim_reduction_size}, attention_size={attention_size}, "
                    f"layer_sizes={layer_sizes}, memory_size={memory_size}")

        logger.info("Initializing cognitive modules")
        perception = PerceptionModule(initial_input_size, conv_params, dim_reduction_size, attention_size, layer_sizes)
        attention = AttentionModule(layer_sizes[-1] + 50, conv_params, dim_reduction_size, attention_size, layer_sizes)
        memory = MemoryModule(8 + 50 + memory_size, conv_params, dim_reduction_size, attention_size, layer_sizes, memory_size)
        reasoning = ReasoningModule(16, conv_params, dim_reduction_size, attention_size, layer_sizes)
        action_selection = ActionSelectionModule(layer_sizes[-1] + 50, conv_params, dim_reduction_size, attention_size, layer_sizes)
        logger.info("Cognitive modules initialized")

        logger.info("Initializing TerminalVisualizer")
        visualizer = TerminalVisualizer()
        logger.info("TerminalVisualizer initialized")

        logger.info("Initializing RealTimeFramework")
        framework = RealTimeFramework()
        logger.info("Adding cognitive_cycle task to framework")
        framework.add_task(cognitive_cycle, interval=0.5, name="cognitive_cycle")

        logger.info("Starting framework")
        await framework.run()

    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        logger.info("Main function completed")

if __name__ == "__main__":
    asyncio.run(main())
