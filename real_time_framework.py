import asyncio
import time
from typing import Callable, Dict, List, Optional

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
        self.loop = asyncio.get_event_loop()
        print("RealTimeFramework initialized")  # Debug print

    def add_task(self, coro: Callable, interval: Optional[float] = None, name: str = ""):
        task = Task(coro, interval, name)
        self.tasks[task.name] = task
        print(f"Task added: {task.name}")  # Debug print

    def remove_task(self, name: str):
        if name in self.tasks:
            del self.tasks[name]
            print(f"Task removed: {name}")  # Debug print

    async def run_task(self, task: Task):
        print(f"Attempting to run task: {task.name}")  # Debug print
        while True:
            if not task.is_running:
                task.is_running = True
                try:
                    await task.coro()
                    print(f"Task completed: {task.name}")  # Debug print
                except Exception as e:
                    print(f"Error in task {task.name}: {e}")
                finally:
                    task.is_running = False
                    task.last_run = time.time()
            
            if task.interval is None:
                print(f"One-time task {task.name} finished")  # Debug print
                break  # This is a one-time task
            
            # Wait for the next interval
            await asyncio.sleep(max(0, task.interval - (time.time() - task.last_run)))

    async def main_loop(self):
        print("Entering main loop")  # Debug print
        while True:
            tasks = []
            for task in self.tasks.values():
                if task.interval is None or time.time() - task.last_run >= task.interval:
                    if not task.is_running:
                        tasks.append(self.run_task(task))
            
            if tasks:
                print(f"Running {len(tasks)} tasks")  # Debug print
                await asyncio.gather(*tasks)
            else:
                await asyncio.sleep(0.1)  # Avoid busy-waiting

    def run(self):
        print("Starting framework run")  # Debug print
        try:
            self.loop.run_until_complete(self.main_loop())
        except KeyboardInterrupt:
            print("Keyboard interrupt received")
        except Exception as e:
            print(f"An error occurred in the framework: {e}")
        finally:
            print("Framework run complete")  # Debug print

# Example usage
async def periodic_task():
    print(f"Periodic task running at {time.time()}")

async def one_time_task():
    print(f"One-time task running at {time.time()}")

if __name__ == "__main__":
    framework = RealTimeFramework()
    
    # Add a periodic task that runs every 2 seconds
    framework.add_task(periodic_task, interval=2, name="periodic")
    
    # Add a one-time task
    framework.add_task(one_time_task, name="one_time")
    
    # Run the framework
    try:
        framework.run()
    except KeyboardInterrupt:
        print("Shutting down...")
