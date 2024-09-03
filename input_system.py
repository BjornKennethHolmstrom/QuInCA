# input_system.py

import sounddevice as sd
import numpy as np
import keyboard
import mouse
from pynput import mouse as pynput_mouse
from pynput import keyboard as pynput_keyboard
from PIL import Image
import mss

class InputSystem:
    def __init__(self, sample_rate=44100, block_size=1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.audio_buffer = np.zeros(block_size, dtype=np.float32)
        
        self.keyboard_events = []
        self.mouse_events = []
        
        # Set up keyboard and mouse listeners
        self.keyboard_listener = pynput_keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.mouse_listener = pynput_mouse.Listener(on_move=self.on_mouse_move, on_click=self.on_mouse_click)
        
        self.keyboard_listener.start()
        self.mouse_listener.start()

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[0]  # Capture the main monitor

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_buffer = indata[:, 0]

    def start_audio_stream(self):
        self.audio_stream = sd.InputStream(samplerate=self.sample_rate, channels=1, 
                                           callback=self.audio_callback, 
                                           blocksize=self.block_size)
        self.audio_stream.start()

    def stop_audio_stream(self):
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()

    def get_audio_data(self):
        return self.audio_buffer.copy()

    def get_screenshot(self):
        screenshot = self.sct.grab(self.monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return np.array(img)

    def on_key_press(self, key):
        self.keyboard_events.append(('press', key))

    def on_key_release(self, key):
        self.keyboard_events.append(('release', key))

    def on_mouse_move(self, x, y):
        self.mouse_events.append(('move', (x, y)))

    def on_mouse_click(self, x, y, button, pressed):
        self.mouse_events.append(('click', (x, y, button, pressed)))

    def get_keyboard_mouse_data(self):
        keyboard_data = self.keyboard_events.copy()
        mouse_data = self.mouse_events.copy()
        self.keyboard_events.clear()
        self.mouse_events.clear()
        return keyboard_data, mouse_data

    def read_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def fetch_internet_data(self, url):
        try:
            response = requests.get(url)
            return response.text
        except Exception as e:
            print(f"Error fetching internet data: {e}")
            return None

    def get_all_input_data(self):
        audio_data = self.get_audio_data()
        keyboard_data, mouse_data = self.get_keyboard_mouse_data()
        screenshot_data = self.get_screenshot()
        
        return {
            'audio': audio_data,
            'keyboard': keyboard_data,
            'mouse': mouse_data,
            'screenshot': screenshot_data,
        }

# Example usage
if __name__ == "__main__":
    input_system = InputSystem()
    input_system.start_audio_stream()
    
    try:
        while True:
            input_data = input_system.get_all_input_data()
            print("Audio data shape:", input_data['audio'].shape)
            print("Keyboard events:", input_data['keyboard'])
            print("Mouse events:", input_data['mouse'])
            print("Screenshot shape:", input_data['screenshot'].shape)
            # Process or pass this data to your perception module
    except KeyboardInterrupt:
        print("Stopping input system...")
    finally:
        input_system.stop_audio_stream()
        input_system.keyboard_listener.stop()
        input_system.mouse_listener.stop()
