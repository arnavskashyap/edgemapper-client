import os
import time
from jetbot import Robot

from .collect_images import collect_images  

class Bot:
    """
    Class dedicated to controlling the JetBot's movement and capturing images.
    Utilizes the collect_images function to save images into the jetbot_images folder.

    Args:
        capture_interval (float): Time interval between image captures in seconds (default: 0.5).
    """

    def __init__(self, capture_interval=0.1, image_count=50):
        # Initialize JetBot components
        self.robot = Robot()
        self.capture_interval = capture_interval
        self.image_count = image_count

    def move_forward(self, speed=0.3, duration=1):
        """Move the JetBot forward and capture images."""
        self.robot.forward(speed)
        time.sleep(duration)
        self.robot.stop()

    def move_backward(self, speed=0.3, duration=1):
        """Move the JetBot backward and capture images."""
        self.robot.backward(speed)
        time.sleep(duration)
        self.robot.stop()

    def turn_left(self, speed=0.3, duration=0.5):
        """Turn the JetBot left and capture images."""
        self.robot.left(speed)
        time.sleep(duration)
        self.robot.stop()

    def turn_right(self, speed=0.3, duration=0.5):
        """Turn the JetBot right and capture images."""
        self.robot.right(speed)
        time.sleep(duration)
        self.robot.stop()

    def _capture_images(self, training_data_path, camera, direction):
        """Call the existing collect_images function to capture images."""
        collect_images(self.image_count, self.capture_interval, training_data_path, camera, self.robot, direction)
        print("Captured images using collect_images.py")

    def stop(self):
        """Stop the JetBot."""
        self.robot.stop()


