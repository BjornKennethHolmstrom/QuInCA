import asyncio
from typing import Callable, Dict, List, Optional
from logger import setup_logger

class Task:
    def __init__(self, coro: Callable, interval: Optional[float] = None, name: str = ""):
        self.coro = coro
        self.interval = interval
        self.name = name or coro.__name__
        self.last_run: float = 0
        self.is_running: bool = False

class RealTimeFramework:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.logger = setup_logger('RealTimeFramework')
        self.logger.debug("RealTimeFramework initialized")

    def add_task(self, coro: Callable, interval: Optional[float] = None, name: str = ""):
        task = Task(coro, interval, name)
        self.tasks[task.name] = task
        self.logger.debug(f"Task added: {task.name}")

    def remove_task(self, name: str):
        if name in self.tasks:
            del self.tasks[name]
            self.logger.debug(f"Task removed: {name}")

    async def run_task(self, task: Task):
        self.logger.debug(f"Attempting to run task: {task.name}")
        while True:
            if not task.is_running:
                task.is_running = True
                try:
                    await task.coro()
                    self.logger.debug(f"Task completed: {task.name}")
                except Exception as e:
                    self.logger.error(f"Error in task {task.name}: {e}", exc_info=True)
                finally:
                    task.is_running = False
                    task.last_run = asyncio.get_event_loop().time()
            
            if task.interval is None:
                self.logger.debug(f"One-time task {task.name} finished")
                break  # This is a one-time task
            
            # Wait for the next interval
            await asyncio.sleep(max(0, task.interval - (asyncio.get_event_loop().time() - task.last_run)))

    async def main_loop(self):
        self.logger.debug("Entering main loop")
        while True:
            tasks = []
            for task in self.tasks.values():
                if task.interval is None or asyncio.get_event_loop().time() - task.last_run >= task.interval:
                    if not task.is_running:
                        tasks.append(self.run_task(task))
            
            if tasks:
                self.logger.debug(f"Running {len(tasks)} tasks")
                await asyncio.gather(*tasks)
            else:
                await asyncio.sleep(0.1)  # Avoid busy-waiting

    async def run(self):
        self.logger.debug("Starting framework run")
        try:
            await self.main_loop()
        except asyncio.CancelledError:
            self.logger.info("Framework run cancelled")
        except Exception as e:
            self.logger.error(f"An error occurred in the framework: {e}", exc_info=True)
        finally:
            self.logger.debug("Framework run complete")

# Example usage
async def periodic_task():
    print(f"Periodic task running at {asyncio.get_event_loop().time()}")

async def one_time_task():
    print(f"One-time task running at {asyncio.get_event_loop().time()}")

if __name__ == "__main__":
    framework = RealTimeFramework()
    
    # Add a periodic task that runs every 2 seconds
    framework.add_task(periodic_task, interval=2, name="periodic")
    
    # Add a one-time task
    framework.add_task(one_time_task, name="one_time")
    
    # Run the framework
    asyncio.run(framework.run())
