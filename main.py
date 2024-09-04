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

async def main():
    logger.info("Starting main function")

    try:
        logger.info("Initializing InputSystem")
        input_system = InputSystem()
        logger.info("InputSystem initialized")

        # Common configuration for all modules
        input_size = 2485  # This is the size of your combined features
        conv_params = {'kernel_size': 3, 'num_filters': 16}
        dim_reduction_size = 128
        attention_size = 64
        layer_sizes = [64, 32, 16, 8]
        memory_size = 100

        logger.info(f"Main configuration: input_size={input_size}, conv_params={conv_params}, "
                    f"dim_reduction_size={dim_reduction_size}, attention_size={attention_size}, "
                    f"layer_sizes={layer_sizes}, memory_size={memory_size}")

        logger.info("Initializing cognitive modules")
        perception = PerceptionModule(input_size, conv_params, dim_reduction_size, attention_size, layer_sizes)
        attention = AttentionModule(layer_sizes[-1] + 50, conv_params, dim_reduction_size, attention_size, layer_sizes)
        memory = MemoryModule(8 + 50 + memory_size, conv_params, dim_reduction_size, attention_size, layer_sizes, memory_size)
        reasoning = ReasoningModule(16, conv_params, dim_reduction_size, attention_size, layer_sizes)
        action_selection = ActionSelectionModule(layer_sizes[-1] + 50, conv_params, dim_reduction_size, attention_size, layer_sizes)
        logger.info("Cognitive modules initialized")

        logger.info("Initializing TerminalVisualizer")
        visualizer = TerminalVisualizer()
        logger.info("TerminalVisualizer initialized")

        # Placeholder for system state
        system_state = np.zeros(50)  # Adjust size as needed

        async def cognitive_cycle():
            logger.debug("Starting cognitive cycle")
            try:
                input_data = input_system.get_all_input_data()
                
                # Perception
                perception_output = await perception.process(input_data)
                logger.debug(f"Perception output shape: {perception_output.shape}")
                
                # Initialize system state if it doesn't exist
                if 'system_state' not in globals():
                    global system_state
                    system_state = np.zeros((1, 50))  # Adjust size as needed, make it 2D
                
                # Attention
                attended_features, attention_weights = await attention.process(perception_output, system_state)
                logger.debug(f"Attended features shape: {attended_features.shape}")
                logger.debug(f"Attention weights shape: {attention_weights.shape}")
                
                # Memory
                placeholder_reasoning_output = np.zeros((1, 50))  # Make it 2D
                retrieved_memories = await memory.process(attended_features, placeholder_reasoning_output)
                logger.debug(f"Retrieved memories shape: {retrieved_memories.shape}")
                
                # Reasoning
                reasoning_output = await reasoning.process(attended_features, retrieved_memories)
                logger.debug(f"Reasoning output shape: {reasoning_output.shape}")
                
                # Action Selection
                selected_action = await action_selection.process(reasoning_output, system_state)
                logger.info(f"Selected action: {selected_action}")
                
                # Update system state (this is a placeholder, replace with actual state update logic)
                system_state = np.random.rand(1, 50)  # Make it 2D
                
                logger.debug("Cognitive cycle completed")
            except Exception as e:
                logger.error(f"Error in cognitive_cycle: {e}", exc_info=True)

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
